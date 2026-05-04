from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import read_jsonl, write_csv, write_jsonl
from src.data.normalization import count_words_vi, is_unknown_or_vague, normalize_answer
from src.data.qa_filtering import (
    LOW_VALUE_QUESTION_TERMS,
    VALID_ANSWER_TYPES,
    VALID_QUESTION_TYPES,
    deduplicate_qa_pairs,
    infer_answer_type,
    infer_question_type,
    infer_question_type_from_text,
    has_sign_anchor,
    is_related_to_signs,
    is_valid_answer,
    is_valid_question,
    mentions_non_sign_object,
    normalize_question_text,
    select_balanced_qa,
)


CSV_FIELDS = [
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


def has_ambiguous_reference(question: str) -> bool:
    key = normalize_question_text(question)
    return "bien nay" in key or " no " in f" {key} "


def has_low_value_detail(question: str) -> bool:
    key = normalize_question_text(question)
    return any(term in key for term in LOW_VALUE_QUESTION_TERMS)


def object_count_from_hint(raw_record: dict[str, Any]) -> int:
    hint = str(raw_record.get("object_hint", ""))
    return hint.count("object_id:")


def requires_specific_evidence(question_type: str, question: str) -> bool:
    key = normalize_question_text(question)
    if question_type in {"sign_type", "location", "color", "shape"}:
        return True
    if question_type == "attribute":
        return True
    if question_type == "negative":
        return False
    if question_type == "yes_no":
        return "bien bao giao thong" not in key
    if question_type == "count":
        return False
    return False


def has_question_type_mismatch(question: str, question_type: str, answer_type: str | None) -> bool:
    expected = infer_question_type_from_text(question)
    if expected == "yes_no" and question_type in {"yes_no", "negative"}:
        return False
    if expected == "count" and question_type == "count" and answer_type == "number":
        return False
    return expected != question_type


def normalize_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            return [value.strip()]
    return []


def make_row(
    raw_record: dict[str, Any],
    qa: dict[str, Any],
    keep: int,
    note: str,
) -> dict[str, Any]:
    question = str(qa.get("question", "")).strip()
    question_type = qa.get("question_type")
    answer_type = qa.get("answer_type") if qa.get("answer_type") in VALID_ANSWER_TYPES else None
    normalize_hint = answer_type
    if question_type == "count" or "bao nhiêu" in question.lower():
        normalize_hint = "number"
    answer = normalize_answer(str(qa.get("answer", "")), normalize_hint)
    inferred_answer_type = infer_answer_type(answer)
    if answer_type is None or inferred_answer_type != "other":
        answer_type = inferred_answer_type

    if question_type not in VALID_QUESTION_TYPES:
        question_type = infer_question_type(question, answer)
    elif question_type == "yes_no" and answer == "Không":
        question_type = "negative"

    return {
        "question_id": "",
        "image_id": raw_record.get("image_id", ""),
        "image_path": raw_record.get("image_path", ""),
        "split": raw_record.get("split", ""),
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "answer_type": answer_type,
        "evidence_object_ids": normalize_evidence(qa.get("evidence_object_ids")),
        "label_source": (
            "rule_based_auto_filtered"
            if raw_record.get("provider") == "dry_run"
            else "gemini_generated_auto_filtered"
        ),
        "keep": keep,
        "corrected_question": "",
        "corrected_answer": "",
        "note": note,
    }


def evaluate_qa(raw_record: dict[str, Any], qa: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    row = make_row(raw_record, qa, keep=1, note="")
    notes: list[str] = []
    hard_fail = False

    question = row["question"]
    answer = row["answer"]

    if not question:
        notes.append("empty_question")
        hard_fail = True
    if not answer:
        notes.append("empty_answer")
        hard_fail = True
    if question and not question.endswith("?"):
        notes.append("question_not_ending_with_question_mark")
        hard_fail = True
    if question and not is_valid_question(question):
        notes.append("invalid_question")
        hard_fail = True
    if answer and not is_valid_answer(answer):
        notes.append("invalid_answer")
        hard_fail = True
    if answer and count_words_vi(answer) > 10:
        notes.append("answer_too_long")
        hard_fail = True
    if answer and is_unknown_or_vague(answer):
        notes.append("vague_or_unknown_answer")
        hard_fail = True
    if question and not is_related_to_signs(question):
        notes.append("possibly_not_traffic_sign_related")
        hard_fail = True
    if question and mentions_non_sign_object(question) and not has_sign_anchor(question):
        notes.append("non_sign_object_question")
        hard_fail = True
    if question and has_low_value_detail(question):
        notes.append("low_value_detail_question")
        hard_fail = True
    if object_count_from_hint(raw_record) > 1 and has_ambiguous_reference(question):
        notes.append("ambiguous_reference_multi_object")
        hard_fail = True
    if has_question_type_mismatch(question, str(row.get("question_type", "")), row.get("answer_type")):
        notes.append("question_type_mismatch")
        hard_fail = True
    if requires_specific_evidence(str(row.get("question_type", "")), question) and not row.get("evidence_object_ids"):
        notes.append("missing_evidence_for_specific_question")
        hard_fail = True

    row["keep"] = 0 if hard_fail else 1
    row["note"] = ";".join(notes)
    return row, not hard_fail


def assign_question_ids(rows_by_image: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for image_id in sorted(rows_by_image):
        for index, row in enumerate(rows_by_image[image_id], start=1):
            row["question_id"] = f"{image_id}_q{index:02d}"
            output_rows.append(row)
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter raw Gemini VQA labels and export review CSV."
    )
    parser.add_argument("--raw", required=True, help="Input raw_qa.jsonl")
    parser.add_argument("--out_jsonl", required=True, help="Output filtered_qa.jsonl")
    parser.add_argument("--out_csv", required=True, help="Output review.csv")
    parser.add_argument("--max_per_image", type=int, default=10)
    parser.add_argument("--min_per_image", type=int, default=3)
    parser.add_argument("--max_yes_no_ratio", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    raw_records = read_jsonl(args.raw)
    logging.info("Loaded %s raw records from %s", len(raw_records), args.raw)

    candidates_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for raw_record in tqdm(raw_records, desc="Filtering QA"):
        if raw_record.get("status") != "ok":
            continue
        image_id = raw_record.get("image_id", "")
        for qa in raw_record.get("qa_pairs") or []:
            row, passed = evaluate_qa(raw_record, qa)
            if passed:
                candidates_by_image[image_id].append(row)
            else:
                rejected_by_image[image_id].append(row)

    selected_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_id, candidates in candidates_by_image.items():
        deduped = deduplicate_qa_pairs(candidates)
        selected = select_balanced_qa(
            deduped,
            max_per_image=args.max_per_image,
            max_yes_no_ratio=args.max_yes_no_ratio,
        )
        if len(selected) < args.min_per_image:
            logging.warning(
                "%s has only %s QA after filtering; min_per_image=%s",
                image_id,
                len(selected),
                args.min_per_image,
            )
        selected_by_image[image_id].extend(selected)

    review_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_id in sorted(set(selected_by_image) | set(rejected_by_image)):
        review_by_image[image_id].extend(selected_by_image.get(image_id, []))
        review_by_image[image_id].extend(rejected_by_image.get(image_id, []))

    review_rows = assign_question_ids(review_by_image)
    filtered_rows = [
        {
            "question_id": row["question_id"],
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "split": row["split"],
            "question": row["question"],
            "answer": row["answer"],
            "question_type": row["question_type"],
            "answer_type": row["answer_type"],
            "evidence_object_ids": row["evidence_object_ids"],
            "label_source": row.get("label_source", "gemini_generated_auto_filtered"),
            "source": "kaggle_vnts",
        }
        for row in review_rows
        if str(row.get("keep")) == "1"
    ]

    csv_rows = []
    for row in review_rows:
        csv_row = dict(row)
        csv_row["evidence_object_ids"] = json.dumps(
            row.get("evidence_object_ids", []), ensure_ascii=False
        )
        csv_rows.append(csv_row)

    write_jsonl(args.out_jsonl, filtered_rows)
    write_csv(args.out_csv, csv_rows, CSV_FIELDS)
    logging.info("Wrote %s filtered QA to %s", len(filtered_rows), args.out_jsonl)
    logging.info("Wrote %s review rows to %s", len(csv_rows), args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
