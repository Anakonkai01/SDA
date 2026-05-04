from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import read_csv, read_jsonl, write_csv
from src.data.normalization import normalize_key


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply safe deterministic fixes to the rule-based review CSV."
    )
    parser.add_argument("--review_csv", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--out_csv", required=True)
    return parser.parse_args()


def parse_evidence(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
    except json.JSONDecodeError:
        pass
    return []


def load_object_map(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        image_id = str(record.get("image_id", "")).strip()
        objects = {str(obj.get("object_id", "")).strip(): obj for obj in record.get("objects") or []}
        mapping[image_id] = objects
    return mapping


def infer_attribute_answer(class_vi: str) -> str:
    text = str(class_vi or "").lower()
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
    if "chỗ quay xe" in text:
        return "Được quay xe"
    if "bến xe buýt" in text:
        return "Chú ý bến xe buýt"
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
    if "đi chậm" in text:
        return "Giảm tốc độ"
    if "gồ giảm tốc" in text:
        return "Giảm tốc độ"
    if "nguy hiểm" in text or "cảnh báo" in text or "chướng ngoại vật" in text:
        return "Giảm tốc độ"
    if "giao nhau" in text:
        return "Chú ý nơi giao nhau"
    if "vòng xuyến" in text:
        return "Đi theo vòng xuyến"
    if "người đi bộ" in text:
        return "Chú ý người đi bộ"
    if "phải đi vòng sang bên phải" in text:
        return "Đi vòng sang bên phải"
    if "giới hạn chiều cao" in text:
        return "Chú ý giới hạn chiều cao"
    if "cấm xe sơ-mi rơ-moóc" in text:
        return "Xe sơ-mi rơ-moóc không được đi vào"
    if "cấm xe hai và ba bánh" in text:
        return "Xe hai và ba bánh không được đi vào"
    if "cấm ô tô khách và ô tô tải" in text:
        return "Ô tô khách và ô tô tải không được đi vào"
    if "biển gộp làn đường theo phương tiện" in text:
        return "Đi đúng làn theo phương tiện"
    return class_vi.strip()


def clean_class_reference(class_vi: str) -> str:
    text = str(class_vi or "").strip()
    lowered = text.lower()
    if lowered.startswith("biển "):
        return text[5:].strip()
    return text


def fix_attribute_label_question(question: str, answer: str, class_vi: str) -> tuple[str, str] | None:
    norm_question = normalize_key(question)
    if "khi gap" not in norm_question and "khi gặp" not in question.lower():
        return None
    if "nguoi lai can chu y gi" not in norm_question and "người lái cần chú ý gì" not in question.lower():
        return None
    class_ref = clean_class_reference(class_vi)
    if not class_ref:
        return None
    fixed_question = f"Khi gặp biển báo {class_ref.lower()}, người lái cần chú ý gì?"
    fixed_answer = infer_attribute_answer(class_vi)
    return fixed_question, fixed_answer


def fix_duplicate_bien(question: str) -> str | None:
    fixed = re.sub(r"\bbiển biển\b", "biển", question, flags=re.IGNORECASE)
    return fixed if fixed != question else None


def main() -> int:
    setup_logging()
    args = parse_args()

    rows = read_csv(args.review_csv)
    object_map = load_object_map(Path(args.objects))
    fixed_attr = 0
    fixed_dup = 0

    for row in rows:
        image_id = str(row.get("image_id", "")).strip()
        evidence_ids = parse_evidence(str(row.get("evidence_object_ids", "")))
        if len(evidence_ids) == 1:
            obj = object_map.get(image_id, {}).get(evidence_ids[0])
            class_vi = str((obj or {}).get("class_vi", "")).strip()
            if class_vi:
                maybe_fixed = fix_attribute_label_question(
                    str(row.get("question", "")),
                    str(row.get("answer", "")),
                    class_vi,
                )
                if maybe_fixed is not None:
                    row["corrected_question"], row["corrected_answer"] = maybe_fixed
                    fixed_attr += 1

        dup_fixed = fix_duplicate_bien(str(row.get("question", "")))
        if dup_fixed is not None:
            row["corrected_question"] = dup_fixed
            fixed_dup += 1

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.out_csv, rows, fieldnames)
    logging.info("Wrote %s rows to %s", len(rows), args.out_csv)
    logging.info("Fixed attribute rows=%s", fixed_attr)
    logging.info("Fixed duplicated 'bien' rows=%s", fixed_dup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
