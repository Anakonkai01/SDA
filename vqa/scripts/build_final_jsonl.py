import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import ensure_dir, read_csv, write_jsonl
from src.data.normalization import count_words_vi, normalize_answer
from src.data.qa_filtering import (
    VALID_ANSWER_TYPES,
    VALID_QUESTION_TYPES,
    infer_answer_type,
    infer_question_type,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def is_keep(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "keep"}


def parse_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in text.split(";") if part.strip()]


def build_record(row: dict[str, str], source: str, label_source: str) -> dict[str, Any]:
    question = (row.get("corrected_question") or row.get("question") or "").strip()
    raw_answer = (row.get("corrected_answer") or row.get("answer") or "").strip()
    question_type = row.get("question_type")
    answer_type = row.get("answer_type") if row.get("answer_type") in VALID_ANSWER_TYPES else None
    normalize_hint = answer_type
    if question_type == "count" or "bao nhiêu" in question.lower():
        normalize_hint = "number"
    answer = normalize_answer(raw_answer, normalize_hint)

    if count_words_vi(answer) > 10:
        raise ValueError(f"answer too long for {row.get('question_id')}: {answer}")
    if not question:
        raise ValueError(f"empty question for {row.get('question_id')}")
    if not answer:
        raise ValueError(f"empty answer for {row.get('question_id')}")

    inferred_answer_type = infer_answer_type(answer)
    if answer_type is None or inferred_answer_type != "other":
        answer_type = inferred_answer_type

    if question_type not in VALID_QUESTION_TYPES:
        question_type = infer_question_type(question, answer)
    elif question_type == "yes_no" and answer == "Không":
        question_type = "negative"

    return {
        "question_id": row.get("question_id", "").strip(),
        "image_id": row.get("image_id", "").strip(),
        "image_path": row.get("image_path", "").strip(),
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "answer_type": answer_type,
        "evidence_object_ids": parse_evidence(row.get("evidence_object_ids")),
        "split": row.get("split", "").strip(),
        "source": source,
        "label_source": label_source,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final train/val/test JSONL annotations from review CSV."
    )
    parser.add_argument("--review_csv", required=True, help="Path to review.csv or reviewed.csv")
    parser.add_argument("--out_dir", required=True, help="Output annotations directory")
    parser.add_argument("--source", default="kaggle_vnts")
    parser.add_argument("--label_source", default="gemini_generated_human_reviewed")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    rows = read_csv(args.review_csv)
    logging.info("Loaded %s rows from %s", len(rows), args.review_csv)

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    seen_question_ids: set[str] = set()

    for row in rows:
        if not is_keep(row.get("keep")):
            continue
        try:
            record = build_record(row, args.source, args.label_source)
            question_id = record["question_id"]
            if not question_id:
                raise ValueError("missing question_id")
            if question_id in seen_question_ids:
                raise ValueError(f"duplicate question_id: {question_id}")
            seen_question_ids.add(question_id)
            split = record["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"invalid split: {split}")
            by_split[split].append(record)
        except Exception as exc:
            errors.append(str(exc))

    if errors:
        for error in errors[:20]:
            logging.error(error)
        if len(errors) > 20:
            logging.error("... and %s more errors", len(errors) - 20)
        return 1

    out_dir = ensure_dir(args.out_dir)
    for split in ["train", "val", "test"]:
        out_path = out_dir / f"{split}.jsonl"
        write_jsonl(out_path, by_split.get(split, []))
        logging.info("Wrote %s records to %s", len(by_split.get(split, [])), out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
