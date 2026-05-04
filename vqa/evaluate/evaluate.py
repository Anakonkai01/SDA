import os
import sys
import argparse
import gc
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import CLIPProcessor, AutoTokenizer
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import ConfigA, ConfigB
from data_utils.dataset import load_vqa_data
from models.model_a import VQAModelA
from models.model_b import VQAModelB
from evaluate.metrics import compute_metrics


def _resolve_image_path(raw_path: str) -> str:
    if raw_path.startswith("data/processed/"):
        return raw_path
    if not raw_path.startswith("/"):
        return os.path.join("data/processed", raw_path)
    return raw_path


def evaluate_model_a(checkpoint_path, decoder_type, test_samples, device,
                     compute_bertscore=False, bertscore_model="xlm-roberta-base"):
    config = ConfigA()
    clip_processor = CLIPProcessor.from_pretrained(config.model.image_encoder)
    phobert_tokenizer = AutoTokenizer.from_pretrained(config.model.text_encoder)

    model = VQAModelA(
        decoder_type=decoder_type,
        vocab_size=config.model.vocab_size,
        dim=config.model.dim,
        clip_dim=config.model.clip_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    predictions = []
    references = []
    total_time = 0

    for sample in tqdm(test_samples, desc=f"Evaluating {decoder_type}"):
        image = Image.open(sample["image"]).convert("RGB")
        pixel_values = clip_processor(
            images=image, return_tensors="pt",
        ).pixel_values.to(device)

        q_enc = phobert_tokenizer(
            sample["question"],
            max_length=config.data.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = q_enc.input_ids.to(device)
        attention_mask = q_enc.attention_mask.to(device)

        start = time.time()
        result = model.generate(pixel_values, input_ids, attention_mask)
        total_time += time.time() - start

        predictions.append(result[0])
        references.append(sample["answer"])

    metrics = compute_metrics(
        predictions, references,
        compute_bertscore=compute_bertscore,
        bertscore_model=bertscore_model,
        device=str(device),
    )
    add_breakdowns(metrics, predictions, references, test_samples)
    metrics["avg_inference_ms"] = (total_time / len(test_samples)) * 1000
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, predictions


def evaluate_model_b(test_samples, lora_path=None, compute_bertscore=False,
                     bertscore_model="xlm-roberta-base", backend="blip",
                     model_name=None, load_in_4bit=False, min_pixels=None,
                     max_pixels=None, max_new_tokens=20, eval_batch_size=1):
    config = ConfigB()
    if model_name is None:
        if backend == "qwen25":
            model_name = config.model.qwen_model_name
        elif backend == "paligemma2":
            model_name = (
                config.model.paligemma_model_name
                if lora_path
                else config.model.paligemma_mix_model_name
            )
        else:
            model_name = config.model.model_name
    backend_labels = {"blip": "BLIP", "qwen25": "Qwen2.5-VL", "paligemma2": "PaliGemma 2"}
    backend_label = backend_labels[backend]

    if lora_path:
        model_b = VQAModelB(
            model_name=model_name, backend=backend,
            load_in_4bit=load_in_4bit,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        model_b.load_lora(lora_path)
        variant = f"B2 ({backend_label} LoRA)"
    else:
        model_b = VQAModelB(
            model_name=model_name, backend=backend,
            load_in_4bit=load_in_4bit,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        variant = f"B1 ({backend_label} zero-shot)"

    # Preload all unique images once (parallel, saves ~10% per eval)
    unique_paths = set()
    for s in test_samples:
        raw = s.get("image") or s.get("image_path") or ""
        if raw:
            unique_paths.add(_resolve_image_path(raw))
    image_cache = {}
    if unique_paths:
        def load_one(path):
            return path, Image.open(path).convert("RGB")
        print(f"Preloading {len(unique_paths)} images...", end="", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(load_one, p) for p in unique_paths]
            for f in as_completed(futures):
                path, img = f.result()
                image_cache[path] = img
        print(f" done")

    orig_load = model_b._load_image
    model_b._load_image = lambda p: image_cache.get(
        _resolve_image_path(p), orig_load(p)
    ).copy()  # .copy() prevents processor mutation

    predictions = []
    references = []
    total_time = 0

    eval_batch_size = max(1, int(eval_batch_size))
    all_batches = [
        test_samples[i:i + eval_batch_size]
        for i in range(0, len(test_samples), eval_batch_size)
    ]

    for batch_samples in tqdm(all_batches, desc=f"Evaluating {variant}"):
        image_paths = [
            _resolve_image_path(s.get("image") or s.get("image_path") or "")
            for s in batch_samples
        ]
        questions = [s["question"] for s in batch_samples]

        start = time.time()
        batch_predictions = model_b.inference_batch(
            image_paths, questions, max_new_tokens=max_new_tokens,
        )
        total_time += time.time() - start

        predictions.extend(batch_predictions)
        references.extend(s["answer"] for s in batch_samples)

    model_b._load_image = orig_load

    metrics = compute_metrics(
        predictions, references,
        compute_bertscore=compute_bertscore,
        bertscore_model=bertscore_model,
        device=str(getattr(model_b, "device", "cpu")),
    )
    add_breakdowns(metrics, predictions, references, test_samples)
    metrics["avg_inference_ms"] = (total_time / len(test_samples)) * 1000
    del model_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, predictions


def build_prediction_rows(model_key, test_samples, predictions):
    rows = []
    for sample, pred in zip(test_samples, predictions):
        rows.append({
            "model": model_key,
            "question_id": sample.get("question_id"),
            "image_id": sample.get("image_id"),
            "image": sample.get("image"),
            "question": sample.get("question"),
            "reference": sample.get("answer"),
            "prediction": pred,
            "question_type": sample.get("question_type", sample.get("type")),
            "answer_type": sample.get("answer_type"),
        })
    return rows


def add_breakdowns(metrics, predictions, references, samples):
    for field in ("question_type", "answer_type"):
        groups = defaultdict(lambda: {"predictions": [], "references": []})
        for pred, ref, sample in zip(predictions, references, samples):
            key = sample.get(field) or sample.get("type") or "unknown"
            groups[key]["predictions"].append(pred)
            groups[key]["references"].append(ref)
        metrics[f"by_{field}"] = {
            key: {
                "total": len(value["references"]),
                **compute_metrics(value["predictions"], value["references"]),
            }
            for key, value in sorted(groups.items())
        }


def select_stratified_samples(samples, limit, seed):
    target = min(int(limit), len(samples))
    if target <= 0:
        return []

    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx, sample in enumerate(samples):
        key = sample.get("question_type", sample.get("type", "unknown"))
        groups[key].append((idx, sample))
    for items in groups.values():
        rng.shuffle(items)

    selected = []
    keys = sorted(groups)
    while len(selected) < target and keys:
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < target:
                selected.append(groups[key].pop()[1])
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def resolve_default_lora(backend):
    if backend == "blip":
        return "checkpoints/model_b2/best_lora"
    if backend == "paligemma2":
        candidates = [
            "checkpoints_b2_paligemma2_10k/model_b2_paligemma2/best_lora",
            "checkpoints/model_b2_paligemma2/best_lora",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    candidates = [
        "checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora",
        "checkpoints/model_b2_qwen25/best_lora",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["a1", "a2", "b1", "b2", "all"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/processed/annotations")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", type=str, choices=["blip", "qwen25", "paligemma2"],
                        default="blip")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--stratified-limit", type=int, default=None,
                        help="Sample a deterministic question_type-balanced subset.")
    parser.add_argument("--stratified-seed", type=int, default=42)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model", type=str, default="xlm-roberta-base")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--predictions-output", type=str, default=None)
    parser.add_argument("--data-split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_samples = load_vqa_data(args.data, args.data_split)
    if args.stratified_limit is not None:
        test_samples = select_stratified_samples(
            test_samples, args.stratified_limit, args.stratified_seed,
        )
    elif args.limit is not None:
        test_samples = test_samples[:args.limit]
    print(f"Loaded test samples: {len(test_samples)}")

    all_results = {}
    all_prediction_rows = []
    backend_tags = {"blip": "BLIP", "qwen25": "Qwen25", "paligemma2": "PaliGemma2"}
    backend_tag = backend_tags[args.backend]

    if args.model in ("a1", "all"):
        ckpt = args.checkpoint or "checkpoints/model_a1/best.pt"
        metrics, predictions = evaluate_model_a(
            ckpt, "lstm", test_samples, device,
            compute_bertscore=args.bertscore,
            bertscore_model=args.bertscore_model,
        )
        all_results["A1_LSTM"] = metrics
        all_prediction_rows.extend(
            build_prediction_rows("A1_LSTM", test_samples, predictions)
        )
        print(f"\nA1 (LSTM): {metrics}")

    if args.model in ("a2", "all"):
        ckpt = args.checkpoint or "checkpoints/model_a2/best.pt"
        metrics, predictions = evaluate_model_a(
            ckpt, "transformer", test_samples, device,
            compute_bertscore=args.bertscore,
            bertscore_model=args.bertscore_model,
        )
        all_results["A2_Transformer"] = metrics
        all_prediction_rows.extend(
            build_prediction_rows("A2_Transformer", test_samples, predictions)
        )
        print(f"\nA2 (Transformer): {metrics}")

    if args.model in ("b1", "all"):
        metrics, predictions = evaluate_model_b(
            test_samples, lora_path=None,
            compute_bertscore=args.bertscore,
            bertscore_model=args.bertscore_model,
            backend=args.backend,
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            eval_batch_size=args.eval_batch_size,
        )
        b1_key = f"B1_{backend_tag}_ZeroShot"
        all_results[b1_key] = metrics
        all_prediction_rows.extend(
            build_prediction_rows(b1_key, test_samples, predictions)
        )
        print(f"\nB1 ({backend_tag} zero-shot): {metrics}")

    if args.model in ("b2", "all"):
        lora = args.lora_path or resolve_default_lora(args.backend)
        metrics, predictions = evaluate_model_b(
            test_samples, lora_path=lora,
            compute_bertscore=args.bertscore,
            bertscore_model=args.bertscore_model,
            backend=args.backend,
            model_name=args.model_name,
            load_in_4bit=args.load_in_4bit,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            eval_batch_size=args.eval_batch_size,
        )
        b2_key = f"B2_{backend_tag}_LoRA"
        all_results[b2_key] = metrics
        all_prediction_rows.extend(
            build_prediction_rows(b2_key, test_samples, predictions)
        )
        print(f"\nB2 ({backend_tag} LoRA): {metrics}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    if args.predictions_output:
        with open(args.predictions_output, "w", encoding="utf-8") as f:
            for row in all_prediction_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {args.predictions_output}")


if __name__ == "__main__":
    main()
