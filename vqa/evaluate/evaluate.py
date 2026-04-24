import os
import sys
import argparse
import json
import time
import torch
from transformers import CLIPProcessor, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import ConfigA, ConfigB
from data_utils.dataset import load_vqa_data
from models.model_a import VQAModelA
from models.model_b import VQAModelB
from evaluate.metrics import compute_metrics


def evaluate_model_a(checkpoint_path, decoder_type, test_samples, device):
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
        from PIL import Image
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

    metrics = compute_metrics(predictions, references)
    metrics["avg_inference_ms"] = (total_time / len(test_samples)) * 1000
    return metrics, predictions


def evaluate_model_b(test_samples, lora_path=None):
    config = ConfigB()

    if lora_path:
        model_b = VQAModelB(model_name=config.model.model_name)
        model_b.load_lora(lora_path)
        variant = "B2 (LoRA)"
    else:
        model_b = VQAModelB(model_name=config.model.model_name)
        variant = "B1 (Zero-shot)"

    predictions = []
    references = []
    total_time = 0

    for sample in tqdm(test_samples, desc=f"Evaluating {variant}"):
        start = time.time()
        pred = model_b.inference(sample["image"], sample["question"])
        total_time += time.time() - start

        predictions.append(pred)
        references.append(sample["answer"])

    metrics = compute_metrics(predictions, references)
    metrics["avg_inference_ms"] = (total_time / len(test_samples)) * 1000
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["a1", "a2", "b1", "b2", "all"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/vqa/vqa_template.json")
    parser.add_argument("--output", type=str, default="results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_samples = load_vqa_data(args.data, "test")

    all_results = {}

    if args.model in ("a1", "all"):
        ckpt = args.checkpoint or "checkpoints/model_a1/best.pt"
        metrics, _ = evaluate_model_a(ckpt, "lstm", test_samples, device)
        all_results["A1_LSTM"] = metrics
        print(f"\nA1 (LSTM): {metrics}")

    if args.model in ("a2", "all"):
        ckpt = args.checkpoint or "checkpoints/model_a2/best.pt"
        metrics, _ = evaluate_model_a(ckpt, "transformer", test_samples, device)
        all_results["A2_Transformer"] = metrics
        print(f"\nA2 (Transformer): {metrics}")

    if args.model in ("b1", "all"):
        metrics, _ = evaluate_model_b(test_samples, lora_path=None)
        all_results["B1_ZeroShot"] = metrics
        print(f"\nB1 (Zero-shot): {metrics}")

    if args.model in ("b2", "all"):
        lora = args.lora_path or "checkpoints/model_b2/best_lora"
        metrics, _ = evaluate_model_b(test_samples, lora_path=lora)
        all_results["B2_LoRA"] = metrics
        print(f"\nB2 (LoRA): {metrics}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
