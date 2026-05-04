from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import read_csv, read_jsonl, write_csv
from src.data.normalization import normalize_key


OUTPUT_FIELDS = [
    "priority",
    "risk_score",
    "review_reason",
    "question_id",
    "image_id",
    "image_path",
    "split",
    "question",
    "answer",
    "question_type",
    "answer_type",
    "evidence_object_ids",
    "keep",
    "corrected_question",
    "corrected_answer",
    "note",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_object_map(objects_path: Path) -> dict[str, dict[str, Any]]:
    object_map: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(objects_path):
        image_id = str(record.get("image_id", ""))
        object_map[image_id] = record
    return object_map


def normalize_evidence(raw_value: str) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return []


def object_count(record: dict[str, Any] | None) -> int:
    if not record:
        return 0
    return len(record.get("objects") or [])


def score_row(row: dict[str, str], object_record: dict[str, Any] | None) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    question = str(row.get("corrected_question") or row.get("question", "")).strip()
    answer = str(row.get("corrected_answer") or row.get("answer", "")).strip()
    note = ""
    if not (str(row.get("corrected_question", "")).strip() or str(row.get("corrected_answer", "")).strip()):
        note = str(row.get("note", "")).strip()
    question_type = str(row.get("question_type", "")).strip()
    evidence = normalize_evidence(str(row.get("evidence_object_ids", "")))
    object_total = object_count(object_record)

    norm_question = normalize_key(question)
    norm_answer = normalize_key(answer)

    if note:
        score += 3
        reasons.append(f"note:{note}")
    if "possibly_not_traffic_sign_related" in note:
        score += 2
    if "bien bien" in norm_question:
        score += 4
        reasons.append("duplicated_word_bien")
    if question_type == "attribute" and object_record:
        class_names = [normalize_key(obj.get("class_vi", "")) for obj in object_record.get("objects") or []]
        if norm_answer in class_names or norm_answer in {"ben xe buyt", "cho quay xe", "bien gop lan duong theo phuong tien"}:
            score += 4
            reasons.append("attribute_answer_looks_like_label")
    if question_type in {"yes_no", "negative"} and not evidence and "trong anh co bien bao giao thong khong" not in norm_question:
        score += 2
        reasons.append("yes_no_without_evidence")
    if question_type == "sign_type" and object_total > 1 and ("o giua" in norm_question or "phia tren" in norm_question or "ben phai" in norm_question or "ben trai" in norm_question):
        duplicated_positions = sum(
            1 for obj in (object_record.get("objects") or [])
            if normalize_key(str(obj.get("relative_position", ""))) in norm_question
        ) if object_record else 0
        if duplicated_positions > 1:
            score += 3
            reasons.append("ambiguous_position_multi_object")
    if question_type == "count" and "bao nhieu bien bao giao thong" not in norm_question and not evidence:
        score += 1
        reasons.append("count_without_evidence")

    return score, reasons


def priority_label(score: int) -> str:
    if score >= 6:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit rule-based review CSV and prioritize rows for manual review.")
    parser.add_argument("--review_csv", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--min_risk_score", type=int, default=3)
    parser.add_argument("--priorities", nargs="*", default=None, help="Filter to these priorities, e.g. high medium")
    parser.add_argument("--splits", nargs="*", default=None, help="Filter to these splits, e.g. test val")
    parser.add_argument(
        "--match_mode",
        choices=["any", "all"],
        default="any",
        help="How to combine priority/split filters when both are set.",
    )
    parser.add_argument(
        "--skip_keep_zero",
        action="store_true",
        help="Skip rows where keep is not truthy.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    rows = read_csv(args.review_csv)
    object_map = load_object_map(Path(args.objects))

    flagged_rows: list[dict[str, Any]] = []
    priority_counts: Counter[str] = Counter()
    for row in rows:
        if args.skip_keep_zero and str(row.get("keep", "")).strip().lower() not in {"1", "true", "yes", "y", "keep"}:
            continue
        image_id = str(row.get("image_id", "")).strip()
        score, reasons = score_row(row, object_map.get(image_id))
        if score < args.min_risk_score:
            continue
        priority = priority_label(score)
        if args.priorities or args.splits:
            split = str(row.get("split", "")).strip()
            matches = []
            if args.priorities:
                matches.append(priority in set(args.priorities))
            if args.splits:
                matches.append(split in set(args.splits))
            if args.match_mode == "any" and not any(matches):
                continue
            if args.match_mode == "all" and matches and not all(matches):
                continue
        priority_counts[priority] += 1
        flagged_rows.append(
            {
                "priority": priority,
                "risk_score": score,
                "review_reason": ";".join(reasons),
                **row,
            }
        )

    flagged_rows.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(str(item.get("priority")), 9),
            -int(item.get("risk_score", 0)),
            str(item.get("image_id", "")),
            str(item.get("question_id", "")),
        )
    )

    write_csv(args.out_csv, flagged_rows, OUTPUT_FIELDS)
    logging.info("Flagged %s rows for priority review", len(flagged_rows))
    for label in ["high", "medium", "low"]:
        logging.info("%s=%s", label, priority_counts.get(label, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
