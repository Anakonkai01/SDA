import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import append_jsonl, read_jsonl
from src.data.normalization import parse_json_object
from src.data.prompt_utils import build_object_hint, load_prompt, render_prompt


QUESTION_TYPES = {
    "yes_no",
    "count",
    "sign_type",
    "color",
    "shape",
    "location",
    "attribute",
    "negative",
}
ANSWER_TYPES = {"yes_no", "number", "other"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_image_path(processed_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return processed_dir / path


def existing_image_ids(path: Path, resume: bool) -> set[str]:
    if not resume or not path.exists():
        return set()
    return {
        str(record.get("image_id", ""))
        for record in read_jsonl(path)
        if record.get("status") == "ok"
    }


def torch_dtype_from_arg(value: str) -> torch.dtype | str:
    if value == "auto":
        return "auto"
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def build_quantization_config(args: argparse.Namespace) -> Any | None:
    if args.quantization == "none":
        return None

    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if args.compute_dtype == "bfloat16" else torch.float16
    if args.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=args.cpu_offload,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if args.quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=args.cpu_offload,
        )
    raise ValueError(f"Unsupported quantization: {args.quantization}")


def parse_device_map(value: str) -> Any:
    if value in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return value
    if value in {"cuda", "cuda:0", "0"}:
        return {"": 0}
    if value in {"cpu", "disk"}:
        return value
    if value.strip().startswith("{"):
        return json.loads(value)
    return value


def build_max_memory(args: argparse.Namespace) -> dict[Any, str] | None:
    max_memory: dict[Any, str] = {}
    if args.max_memory_gpu:
        max_memory[0] = args.max_memory_gpu
    if args.max_memory_cpu:
        max_memory["cpu"] = args.max_memory_cpu
    return max_memory or None


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    logging.info("Loading processor: %s", args.model)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    kwargs: dict[str, Any] = {
        "device_map": parse_device_map(args.device_map),
        "torch_dtype": torch_dtype_from_arg(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    max_memory = build_max_memory(args)
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if args.attn_implementation != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    if args.offload_dir:
        offload_dir = Path(args.offload_dir)
        offload_dir.mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = str(offload_dir)

    quantization_config = build_quantization_config(args)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    logging.info(
        "Loading model: %s | quantization=%s | device_map=%s",
        args.model,
        args.quantization,
        kwargs["device_map"],
    )
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    return model, processor


def model_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_messages(prompt: str, image_path: Path, image_mode: str) -> list[dict[str, Any]]:
    if image_mode == "pil":
        image = Image.open(image_path).convert("RGB")
        image_part = {"type": "image", "image": image}
    elif image_mode == "path":
        image_part = {"type": "image", "image": str(image_path)}
    elif image_mode == "url":
        image_part = {"type": "image", "url": image_path.resolve().as_uri()}
    else:
        raise ValueError(f"Unsupported image_mode: {image_mode}")

    return [
        {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def response_to_text(processor: Any, response: str) -> str:
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(response)
            if isinstance(parsed, str):
                return parsed
            if isinstance(parsed, dict):
                for key in ("content", "text", "response", "answer"):
                    if isinstance(parsed.get(key), str):
                        return parsed[key]
                return json.dumps(parsed, ensure_ascii=False)
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False)
        except Exception as exc:
            logging.warning("processor.parse_response failed: %s", exc)
    return response


def clean_qa_pairs(raw_pairs: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_pairs, list):
        raise ValueError("qa_pairs is not a list")

    cleaned: list[dict[str, Any]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        question_type = item.get("question_type") or "sign_type"
        answer_type = item.get("answer_type") or "other"
        evidence = item.get("evidence_object_ids") or []
        if question_type not in QUESTION_TYPES:
            question_type = "sign_type"
        if answer_type not in ANSWER_TYPES:
            answer_type = "other"
        if not isinstance(evidence, list):
            evidence = []
        cleaned.append(
            {
                "question_type": question_type,
                "question": str(item.get("question", "")).strip(),
                "answer": str(item.get("answer", "")).strip(),
                "answer_type": answer_type,
                "evidence_object_ids": [str(value) for value in evidence],
            }
        )
    return cleaned


def parse_qa_response(text: str) -> list[dict[str, Any]]:
    parsed = parse_json_object(text)
    if "qa_pairs" not in parsed:
        raise ValueError("JSON response missing key: qa_pairs")
    return clean_qa_pairs(parsed["qa_pairs"])


@torch.inference_mode()
def generate_raw_response(
    model: Any,
    processor: Any,
    prompt: str,
    image_path: Path,
    args: argparse.Namespace,
) -> str:
    messages = build_messages(prompt, image_path, args.image_mode)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = inputs.to(model_input_device(model))
    input_len = inputs["input_ids"].shape[-1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    outputs = model.generate(**inputs, **generation_kwargs)
    decoded = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return response_to_text(processor, decoded)


def make_failed_record(
    record: dict[str, Any],
    object_hint: str,
    args: argparse.Namespace,
    error: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    return {
        "image_id": record.get("image_id", ""),
        "image_path": record.get("image_path", ""),
        "split": record.get("split", ""),
        "provider": "local_gemma4",
        "model": args.model,
        "object_hint": object_hint,
        "qa_pairs": [],
        "raw_response": "",
        "status": "failed",
        "error": error,
        "elapsed_sec": round(elapsed_sec, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def generate_record(
    model: Any,
    processor: Any,
    record: dict[str, Any],
    prompt_template: str,
    processed_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    object_hint = build_object_hint(record)
    prompt = render_prompt(prompt_template, object_hint, args.num_questions)
    image_path = resolve_image_path(processed_dir, str(record.get("image_path", "")))
    start = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        raw_response = generate_raw_response(model, processor, prompt, image_path, args)
        qa_pairs = parse_qa_response(raw_response)
        elapsed = time.perf_counter() - start
        peak_vram_gb = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else None
        )
        return {
            "image_id": record.get("image_id", ""),
            "image_path": record.get("image_path", ""),
            "split": record.get("split", ""),
            "provider": "local_gemma4",
            "model": args.model,
            "object_hint": object_hint,
            "qa_pairs": qa_pairs,
            "raw_response": raw_response,
            "status": "ok",
            "error": None,
            "elapsed_sec": round(elapsed, 3),
            "peak_vram_gb": round(peak_vram_gb, 3) if peak_vram_gb is not None else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return make_failed_record(record, object_hint, args, str(exc), elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test local Gemma 4 VLM labeling on a few VQA images."
    )
    parser.add_argument("--objects", default="data/processed/metadata/objects.jsonl")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--prompt", default="prompts/vqa_traffic_vi.txt")
    parser.add_argument("--out", default="data/processed/review/raw_qa_gemma4_local_smoke.jsonl")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Splits to process or all")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--num_questions", type=int, default=12)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--compute_dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_memory_gpu", default=None, help='Example: "15GiB"')
    parser.add_argument("--max_memory_cpu", default=None, help='Example: "24GiB"')
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--offload_dir", default=".cache/gemma4_offload")
    parser.add_argument("--attn_implementation", default="auto")
    parser.add_argument("--image_mode", choices=["pil", "path", "url"], default="pil")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        logging.error("CUDA is not available. Use --allow_cpu only for tiny dry experiments.")
        return 2

    objects_path = Path(args.objects)
    processed_dir = Path(args.processed_dir)
    out_path = Path(args.out)
    records = read_jsonl(objects_path)
    requested_splits = set(args.splits)
    if "all" not in requested_splits:
        records = [record for record in records if record.get("split") in requested_splits]
    if args.limit is not None:
        records = records[: args.limit]

    skip_ids = existing_image_ids(out_path, args.resume)
    if skip_ids:
        logging.info("Resume enabled: skipping %s completed image_ids", len(skip_ids))

    logging.info("Smoke test records=%s | out=%s", len(records), out_path)
    prompt_template = load_prompt(args.prompt)
    model, processor = load_model_and_processor(args)

    failures = 0
    written = 0
    for record in tqdm(records, desc="Gemma 4 local"):
        image_id = str(record.get("image_id", ""))
        if args.resume and image_id in skip_ids:
            continue
        output_record = generate_record(
            model,
            processor,
            record,
            prompt_template,
            processed_dir,
            args,
        )
        append_jsonl(out_path, output_record)
        written += 1
        if output_record.get("status") != "ok":
            failures += 1
            logging.error("%s failed: %s", image_id, output_record.get("error"))
        else:
            logging.info(
                "%s ok | qa=%s | %.2fs | peak_vram=%s GB",
                image_id,
                len(output_record.get("qa_pairs") or []),
                output_record.get("elapsed_sec", 0.0),
                output_record.get("peak_vram_gb"),
            )

    logging.info("Done. written=%s failures=%s", written, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
