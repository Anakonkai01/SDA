from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

NEGATIVE_CLASS_CANDIDATES = [
    "cấm rẽ trái",
    "cấm rẽ phải",
    "cấm quay đầu",
    "cấm rẽ trái và quay đầu xe",
    "cấm rẽ phải và quay đầu",
    "cấm đi ngược chiều",
    "cấm dừng và đỗ xe",
    "cấm đỗ xe",
    "cấm ô tô",
    "cấm xe tải",
    "cấm mô tô và xe máy",
    "giới hạn tốc độ 50 km/h",
    "giới hạn tốc độ 60 km/h",
    "bến xe buýt",
    "đường người đi bộ cắt ngang",
    "nơi giao nhau chạy theo vòng xuyến",
    "đường một chiều",
    "trẻ em",
]


def is_rate_limit_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return "http 429" in lowered or "rate-limit" in lowered or "rate limited" in lowered


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        return


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_existing_ids(raw_path: Path, resume: bool, skip_failed: bool) -> set[str]:
    if not resume or not raw_path.exists():
        return set()
    existing: set[str] = set()
    for record in read_jsonl(raw_path):
        status = record.get("status")
        if status == "ok" or (skip_failed and status == "failed"):
            existing.add(record.get("image_id", ""))
    return existing


def resolve_image_path(processed_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return processed_dir / path


def requires_specific_evidence(question_type: str, question: str) -> bool:
    key = re.sub(r"[^\w\s]", "", str(question or "").lower())
    if question_type in {"sign_type", "location", "color", "shape", "attribute"}:
        return True
    if question_type in {"yes_no", "negative"}:
        return "bien bao giao thong" not in key
    if question_type == "count":
        return "bao nhieu bien bao giao thong" not in key
    return False


def clean_qa_pairs(raw_pairs: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_pairs, list):
        raise ValueError("qa_pairs is not a list")

    cleaned: list[dict[str, Any]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question and not answer:
            continue
        question_type = item.get("question_type") or "sign_type"
        answer_type = item.get("answer_type") or "other"
        evidence = item.get("evidence_object_ids") or []
        if not isinstance(evidence, list):
            evidence = []
        if question_type not in QUESTION_TYPES:
            question_type = "sign_type"
        if answer_type not in ANSWER_TYPES:
            answer_type = "other"
        evidence = [str(e) for e in evidence if str(e).strip()]
        if requires_specific_evidence(question_type, question) and not evidence:
            continue
        cleaned.append({
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "evidence_object_ids": evidence,
        })
    return cleaned


def parse_qa_response(text: str) -> list[dict[str, Any]]:
    parsed = parse_json_object(text)
    if "qa_pairs" not in parsed:
        raise ValueError("JSON response missing key: qa_pairs")
    return clean_qa_pairs(parsed["qa_pairs"])


def infer_color(class_vi: str) -> str:
    text = class_vi.lower()
    if "cấm" in text or "tốc độ" in text or "hạn chế" in text:
        return "Đỏ và trắng"
    if "cảnh báo" in text or "nguy hiểm" in text or "giao nhau" in text:
        return "Vàng và đen"
    if "chỉ dẫn" in text or "một chiều" in text or "quay xe" in text:
        return "Xanh và trắng"
    return "Đỏ và trắng"


def answer_text(text: str, max_words: int = 10) -> str:
    text = str(text or "").replace("*", "").strip()
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.split(",")[0].strip()
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text[:1].upper() + text[1:] if text else ""


def infer_shape(class_vi: str) -> str:
    text = class_vi.lower()
    if "cảnh báo" in text or "nguy hiểm" in text:
        return "Hình tam giác"
    if "phụ" in text or "thuyết minh" in text:
        return "Hình chữ nhật"
    return "Hình tròn"


def infer_attribute_answer(class_vi: str) -> str:
    text = class_vi.lower()
    speed_match = re.search(r"\b(\d{1,3})\s*km\s*/?\s*h\b", text)
    if speed_match:
        return f"{int(speed_match.group(1))} km/h"
    if "camera giám sát" in text:
        return "Chú ý có camera giám sát"
    if "đường một chiều" in text:
        return "Đi theo một chiều"
    if "trẻ em" in text:
        return "Chú ý trẻ em"
    if "các xe chỉ được rẽ trái" in text:
        return "Chỉ được rẽ trái"
    if "cấm xe tải" in text:
        return "Xe tải không được đi vào"
    if "cấm mô tô và xe máy" in text:
        return "Mô tô và xe máy không được đi vào"
    if "rẽ trái" in text and "cấm" in text:
        return "Không được rẽ trái"
    if "rẽ phải" in text and "cấm" in text:
        return "Không được rẽ phải"
    if "quay đầu" in text and "cấm" in text:
        return "Không được quay đầu"
    if "ngược chiều" in text:
        return "Không được đi ngược chiều"
    if "dừng" in text or "đỗ" in text:
        return "Không được dừng đỗ"
    if "cấm" in text:
        return answer_text(class_vi)
    if "giao nhau" in text:
        return "Chú ý nơi giao nhau"
    if "nguy hiểm" in text or "cảnh báo" in text or "đi chậm" in text:
        return "Giảm tốc độ"
    if "bến xe buýt" in text:
        return "Bến xe buýt"
    if "vòng xuyến" in text:
        return "Đi theo vòng xuyến"
    if "người đi bộ" in text:
        return "Chú ý người đi bộ"
    return answer_text(class_vi)


def get_obj_answer(obj: dict[str, Any]) -> str:
    return answer_text(obj.get("class_vi") or "biển báo giao thông")


def get_obj_group(obj: dict[str, Any]) -> str:
    return answer_text(obj.get("group") or "biển báo")


def get_obj_color(obj: dict[str, Any]) -> str:
    return obj.get("color_hint") or infer_color(obj.get("class_vi") or "")


def get_obj_shape(obj: dict[str, Any]) -> str:
    return obj.get("shape") or infer_shape(obj.get("class_vi") or "")


def tokenize_label(text: str) -> set[str]:
    return set(re.findall(r"\w+", str(text or "").lower()))


def labels_are_related(candidate: str, present_label: str) -> bool:
    candidate_norm = re.sub(r"\s+", " ", str(candidate or "").lower()).strip()
    present_norm = re.sub(r"\s+", " ", str(present_label or "").lower()).strip()
    if not candidate_norm or not present_norm:
        return False
    if candidate_norm in present_norm or present_norm in candidate_norm:
        return True

    candidate_tokens = tokenize_label(candidate_norm)
    present_tokens = tokenize_label(present_norm)
    if not candidate_tokens or not present_tokens:
        return False
    return candidate_tokens.issubset(present_tokens) or present_tokens.issubset(candidate_tokens)


def choose_absent_class(objects: list[dict[str, Any]]) -> str:
    present = [str(obj.get("class_vi", "")).lower() for obj in objects]
    for candidate in NEGATIVE_CLASS_CANDIDATES:
        if not any(labels_are_related(candidate, label) for label in present):
            return candidate
    return "cấm rẽ trái"


def position_phrase(position: str) -> str:
    phrase = str(position or "ở giữa").lower()
    if phrase.startswith("ở "):
        return phrase
    return f"ở {phrase}"


def make_mock_qa(record: dict[str, Any], num_questions: int) -> list[dict[str, Any]]:
    objects = record.get("objects") or []
    first = objects[0] if objects else {}
    second = objects[1] if len(objects) > 1 else first
    third = objects[2] if len(objects) > 2 else second

    first_id = first.get("object_id")
    second_id = second.get("object_id")
    third_id = third.get("object_id")
    first_evidence = [first_id] if first_id else []
    second_evidence = [second_id] if second_id else first_evidence
    third_evidence = [third_id] if third_id else second_evidence
    all_evidence = [str(obj["object_id"]) for obj in objects if obj.get("object_id")]

    class_vi = get_obj_answer(first)
    second_class_vi = get_obj_answer(second)
    third_class_vi = get_obj_answer(third)
    position = first.get("relative_position") or "? gi?a"
    second_position = second.get("relative_position") or position
    third_position = third.get("relative_position") or second_position
    position_text = position_phrase(position)
    second_position_text = position_phrase(second_position)
    third_position_text = position_phrase(third_position)
    count = str(len(objects))

    color = get_obj_color(first)
    shape = get_obj_shape(first)
    second_color = get_obj_color(second)
    second_shape = get_obj_shape(second)
    group = get_obj_group(first)
    second_group = get_obj_group(second)
    third_group = get_obj_group(third)
    attribute_answer = infer_attribute_answer(first.get("class_vi") or class_vi)
    second_attribute_answer = infer_attribute_answer(second.get("class_vi") or second_class_vi)
    absent_class = choose_absent_class(objects)

    group_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    for obj in objects:
        group_name = get_obj_group(obj)
        class_name = get_obj_answer(obj)
        group_counts[group_name] = group_counts.get(group_name, 0) + 1
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    count_group = max(group_counts, key=group_counts.get) if group_counts else "bi?n b?o"

    candidates = [
        ("yes_no", "Trong ?nh c? bi?n b?o giao th?ng kh?ng?", "C?", "yes_no", []),
        ("count", "C? bao nhi?u bi?n b?o giao th?ng trong ?nh?", count, "number", all_evidence),
        ("sign_type", f"Bi?n b?o {position_text} l? g??", class_vi, "other", first_evidence),
        ("sign_type", f"Bi?n b?o {position_text} thu?c nh?m n?o?", group, "other", first_evidence),
        ("location", f"{class_vi} n?m ? ??u?", position, "other", first_evidence),
        ("color", f"{class_vi} c? m?u g??", color, "other", first_evidence),
        ("shape", f"{class_vi} c? h?nh g??", shape, "other", first_evidence),
        ("attribute", f"Khi g?p {class_vi}, ng??i l?i c?n ch? ? g??", attribute_answer, "other", first_evidence),
        ("negative", f"Trong ?nh c? bi?n {absent_class} kh?ng?", "Kh?ng", "yes_no", []),
        ("yes_no", f"Trong ?nh c? bi?n {class_vi.lower()} kh?ng?", "C?", "yes_no", first_evidence),
        ("count", f"C? bao nhi?u {count_group.lower()} trong ?nh?", str(group_counts.get(count_group, 0)), "number", all_evidence),
        ("sign_type", f"Bi?n b?o {second_position_text} l? g??", second_class_vi, "other", second_evidence),
        ("sign_type", f"Bi?n b?o {second_position_text} thu?c nh?m n?o?", second_group, "other", second_evidence),
        ("count", f"Trong ?nh c? bao nhi?u bi?n {class_vi.lower()}?", str(class_counts.get(class_vi, 0)), "number", all_evidence),
        ("sign_type", f"T?n c?a bi?n b?o {position_text} l? g??", class_vi, "other", first_evidence),
        ("color", f"Bi?n b?o {position_text} c? m?u g??", color, "other", first_evidence),
        ("shape", f"Bi?n b?o {position_text} c? h?nh g??", shape, "other", first_evidence),
        ("attribute", f"Bi?n {class_vi.lower()} nh?c ng??i l?i ?i?u g??", attribute_answer, "other", first_evidence),
        ("location", f"Bi?n b?o {second_class_vi.lower()} n?m ? ??u?", second_position, "other", second_evidence),
        ("color", f"{second_class_vi} c? m?u g??", second_color, "other", second_evidence),
        ("shape", f"{second_class_vi} c? h?nh g??", second_shape, "other", second_evidence),
        ("attribute", f"Khi g?p {second_class_vi}, ng??i l?i c?n ch? ? g??", second_attribute_answer, "other", second_evidence),
        ("yes_no", f"Bi?n {class_vi.lower()} c? trong ?nh kh?ng?", "C?", "yes_no", first_evidence),
        ("sign_type", f"Bi?n b?o {third_position_text} l? g??", third_class_vi, "other", third_evidence),
        ("sign_type", f"Bi?n b?o {third_position_text} thu?c nh?m n?o?", third_group, "other", third_evidence),
    ]
    return [
        {
            "question_type": qtype,
            "question": question,
            "answer": answer,
            "answer_type": atype,
            "evidence_object_ids": ev,
        }
        for qtype, question, answer, atype, ev in candidates[:num_questions]
    ]


def generate_with_google_genai(
    api_key: str,
    model_name: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_output_tokens: int,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)
    response = client.models.generate_content(
        model=model_name,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def generate_with_google_generativeai(
    api_key: str,
    model_name: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_output_tokens: int,
) -> str:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    image = Image.open(image_path).convert("RGB")
    response = model.generate_content(
        [prompt, image],
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
        },
    )
    return response.text or ""


def call_gemini(
    api_key: str,
    model_name: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_output_tokens: int,
) -> str:
    try:
        return generate_with_google_genai(
            api_key, model_name, prompt, image_path, temperature, max_output_tokens
        )
    except ImportError:
        return generate_with_google_generativeai(
            api_key, model_name, prompt, image_path, temperature, max_output_tokens
        )


def encode_image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_openrouter_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"OpenRouter response has no choices: {response}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ]
        return "\n".join(part for part in text_parts if part)
    raise ValueError(f"Unsupported OpenRouter content type: {type(content).__name__}")


def call_openrouter(
    api_key: str,
    model_name: str,
    prompt: str,
    image_path: Path,
    temperature: float,
    max_output_tokens: int,
    base_url: str,
    site_url: str | None,
    app_name: str | None,
    timeout: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_data_url(image_path)},
                    },
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    parsed = json.loads(body)
    return extract_openrouter_content(parsed)


def generate_record(
    record: dict[str, Any],
    processed_dir: Path,
    prompt_template: str,
    args: argparse.Namespace,
    api_key: str | None,
) -> dict[str, Any]:
    object_hint = build_object_hint(record)
    prompt = render_prompt(prompt_template, object_hint, args.num_questions)
    image_path = record.get("image_path", "")
    image_full_path = resolve_image_path(processed_dir, image_path)

    base = {
        "image_id": record.get("image_id", ""),
        "image_path": image_path,
        "split": record.get("split", ""),
        "provider": "dry_run" if args.dry_run else args.provider,
        "model": "rule_based" if args.dry_run else args.model,
        "object_hint": object_hint,
    }

    if args.dry_run:
        qa_pairs = make_mock_qa(record, args.num_questions)
        return {
            **base,
            "qa_pairs": qa_pairs,
            "raw_response": json.dumps({"qa_pairs": qa_pairs}, ensure_ascii=False),
            "status": "ok",
            "error": None,
        }

    if args.provider not in {"gemini", "openrouter"}:
        raise ValueError(f"Unsupported provider: {args.provider}")
    if not api_key:
        env_name = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENROUTER_API_KEY"
        raise RuntimeError(f"Missing {env_name}. Set it or use --dry_run.")
    if not image_full_path.exists():
        raise FileNotFoundError(f"Image not found: {image_full_path}")

    last_error: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            if args.provider == "gemini":
                raw_response = call_gemini(
                    api_key,
                    args.model,
                    prompt,
                    image_full_path,
                    args.temperature,
                    args.max_output_tokens,
                )
            else:
                raw_response = call_openrouter(
                    api_key,
                    args.model,
                    prompt,
                    image_full_path,
                    args.temperature,
                    args.max_output_tokens,
                    args.openrouter_base_url,
                    args.openrouter_site_url,
                    args.openrouter_app_name,
                    args.timeout,
                )
            qa_pairs = parse_qa_response(raw_response)
            return {
                **base,
                "qa_pairs": qa_pairs,
                "raw_response": raw_response,
                "status": "ok",
                "error": None,
            }
        except Exception as exc:
            last_error = exc
            logging.warning(
                "Attempt %s/%s failed for %s: %s",
                attempt,
                args.retries,
                record.get("image_id", ""),
                exc,
            )
            if attempt < args.retries:
                sleep_seconds = (
                    args.rate_limit_sleep
                    if args.provider == "openrouter" and is_rate_limit_text(str(exc))
                    else args.retry_delay
                )
                logging.info("Sleeping %.1fs before retry", sleep_seconds)
                time.sleep(sleep_seconds)

    return {
        **base,
        "qa_pairs": [],
        "raw_response": "",
        "status": "failed",
        "error": str(last_error) if last_error else "unknown error",
    }


def write_failed_record(path: Path | None, raw_record: dict[str, Any]) -> None:
    if path is None:
        return
    append_jsonl(path, {
        "image_id": raw_record.get("image_id", ""),
        "image_path": raw_record.get("image_path", ""),
        "split": raw_record.get("split", ""),
        "error": raw_record.get("error", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese traffic-sign VQA labels from objects.jsonl."
    )
    parser.add_argument("--objects", required=True, help="Path to metadata/objects.jsonl")
    parser.add_argument("--processed_dir", required=True, help="Root directory for processed data")
    parser.add_argument("--prompt", required=True, help="Prompt template path")
    parser.add_argument("--out", required=True, help="Output raw_qa.jsonl path")
    parser.add_argument("--failed_out", default=None, help="Output failed_generation.jsonl path")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process or all")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openrouter"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_output_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--openrouter_base_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter_site_url", default=None)
    parser.add_argument("--openrouter_app_name", default="SDA Traffic Sign VQA")
    parser.add_argument("--num_questions", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between successful API calls")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--rate_limit_sleep", type=float, default=60.0)
    parser.add_argument("--stop_on_rate_limit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip images already generated with status=ok")
    parser.add_argument("--skip_failed", action="store_true", help="Also skip failed records when resuming")
    parser.add_argument("--dry_run", action="store_true", help="Generate deterministic mock QA without API")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    load_dotenv_if_available()
    args = parse_args()

    objects_path = Path(args.objects)
    processed_dir = Path(args.processed_dir)
    out_path = Path(args.out)
    failed_out = Path(args.failed_out) if args.failed_out else None

    records = read_jsonl(objects_path)
    logging.info("Loaded %s object records from %s", len(records), objects_path)

    requested_splits = set(args.splits)
    if "all" not in requested_splits:
        records = [r for r in records if r.get("split") in requested_splits]
    if args.limit is not None:
        records = records[:args.limit]

    logging.info("Processing %s images for splits: %s", len(records), ", ".join(args.splits))
    existing_ids = get_existing_ids(out_path, args.resume, args.skip_failed)
    if existing_ids:
        logging.info("Resume enabled: skipping %s existing image_ids", len(existing_ids))

    prompt_template = load_prompt(args.prompt)
    api_key = os.getenv("GEMINI_API_KEY") if args.provider == "gemini" else os.getenv("OPENROUTER_API_KEY")
    if not args.dry_run and not api_key:
        env_name = "GEMINI_API_KEY" if args.provider == "gemini" else "OPENROUTER_API_KEY"
        logging.error("Missing %s. Set it or rerun with --dry_run.", env_name)
        return 2

    written = 0
    failed = 0
    for record in tqdm(records, desc="Generating QA"):
        image_id = record.get("image_id", "")
        if args.resume and image_id in existing_ids:
            continue

        raw_record = generate_record(record, processed_dir, prompt_template, args, api_key)
        append_jsonl(out_path, raw_record)
        written += 1

        status = raw_record.get("status")
        if status == "failed":
            failed += 1
            write_failed_record(failed_out, raw_record)
            if args.stop_on_rate_limit and is_rate_limit_text(str(raw_record.get("error", ""))):
                logging.error("Stopping early because the provider is rate-limited.")
                return 3
        else:
            logging.info(
                "%s | split=%s | objects=%s | qa=%s",
                image_id,
                record.get("split", ""),
                len(record.get("objects") or []),
                len(raw_record.get("qa_pairs") or []),
            )
            if args.delay:
                time.sleep(args.delay)

    logging.info("Done. Wrote %s raw records, failed=%s", written, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
