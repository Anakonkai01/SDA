import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import ensure_parent, read_jsonl
from src.data.normalization import count_words_vi
from src.data.qa_filtering import VALID_ANSWER_TYPES, VALID_QUESTION_TYPES


REQUIRED_QUESTION_TYPES = {
    "yes_no",
    "count",
    "sign_type",
    "color",
    "shape",
    "location",
    "attribute",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_annotations(annotations_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        path = annotations_dir / f"{split}.jsonl"
        for record in read_jsonl(path):
            record.setdefault("split", split)
            records.append(record)
    return records


def load_object_ids(objects_path: Path) -> set[str]:
    object_ids: set[str] = set()
    for record in read_jsonl(objects_path):
        for obj in record.get("objects") or []:
            object_id = obj.get("object_id")
            if object_id:
                object_ids.add(str(object_id))
    return object_ids


def resolve_image_path(processed_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return processed_dir / path


def compute_report(
    records: list[dict[str, Any]],
    object_ids: set[str],
    processed_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    question_ids: set[str] = set()
    image_to_split: dict[str, str] = {}
    qa_by_split: Counter[str] = Counter()
    images_by_split: dict[str, set[str]] = defaultdict(set)
    qa_per_image: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()

    for record in records:
        question_id = record.get("question_id", "")
        image_id = record.get("image_id", "")
        split = record.get("split", "")
        question_type = record.get("question_type", "")
        answer_type = record.get("answer_type", "")
        answer = str(record.get("answer", "")).strip()
        image_path = str(record.get("image_path", "")).strip()

        if not question_id:
            errors.append("missing question_id")
        elif question_id in question_ids:
            errors.append(f"duplicate question_id: {question_id}")
        question_ids.add(question_id)

        if not image_id:
            errors.append(f"{question_id}: missing image_id")
        if split not in {"train", "val", "test"}:
            errors.append(f"{question_id}: invalid split {split}")

        previous_split = image_to_split.get(image_id)
        if previous_split and previous_split != split:
            errors.append(f"image_id appears in multiple splits: {image_id}")
        if image_id:
            image_to_split[image_id] = split
            qa_per_image[image_id] += 1
            images_by_split[split].add(image_id)

        qa_by_split[split] += 1
        question_type_counts[question_type] += 1
        answer_type_counts[answer_type] += 1

        if question_type not in VALID_QUESTION_TYPES:
            errors.append(f"{question_id}: invalid question_type {question_type}")
        if answer_type not in VALID_ANSWER_TYPES:
            errors.append(f"{question_id}: invalid answer_type {answer_type}")
        if not record.get("question"):
            errors.append(f"{question_id}: empty question")
        if not answer:
            errors.append(f"{question_id}: empty answer")
        if count_words_vi(answer) > 10:
            errors.append(f"{question_id}: answer > 10 words")
        if image_path and not resolve_image_path(processed_dir, image_path).exists():
            errors.append(f"{question_id}: image_path not found: {image_path}")

        for object_id in record.get("evidence_object_ids") or []:
            if object_id not in object_ids:
                errors.append(f"{question_id}: unknown evidence_object_id {object_id}")

    total_images = len(image_to_split)
    total_qa = len(records)
    yes_no_ratio = (
        (question_type_counts.get("yes_no", 0) + question_type_counts.get("negative", 0)) / total_qa
        if total_qa else 0.0
    )

    if qa_by_split.get("train", 0) < 2000:
        errors.append(f"train QA < 2000: {qa_by_split.get('train', 0)}")
    if total_images < 200:
        errors.append(f"total unique images < 200: {total_images}")
    if qa_by_split.get("test", 0) < 50:
        errors.append(f"test QA < 50: {qa_by_split.get('test', 0)}")
    for image_id, count in qa_per_image.items():
        if count < 3:
            errors.append(f"{image_id}: QA/image < 3")
    if yes_no_ratio > 0.40:
        errors.append(f"yes_no ratio > 0.40: {yes_no_ratio:.3f}")

    missing_types = sorted(REQUIRED_QUESTION_TYPES - set(question_type_counts))
    if missing_types:
        errors.append(f"missing required question types: {missing_types}")

    split_image_counts = {split: len(images_by_split.get(split, set())) for split in ["train", "val", "test"]}
    if total_images:
        train_ratio = split_image_counts["train"] / total_images
        val_ratio = split_image_counts["val"] / total_images
        test_ratio = split_image_counts["test"] / total_images
        if abs(train_ratio - 0.8) > 0.08 or abs(val_ratio - 0.1) > 0.05 or abs(test_ratio - 0.1) > 0.05:
            warnings.append(
                f"image split ratio not close to 80/10/10: "
                f"train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}"
            )

    test_records = [r for r in records if r.get("split") == "test"]
    if test_records and not any(r.get("question_type") == "negative" for r in test_records):
        warnings.append("test split has no negative/adversarial question")

    qa_counts = list(qa_per_image.values())
    qa_per_image_report = {
        "min": min(qa_counts) if qa_counts else 0,
        "mean": (sum(qa_counts) / len(qa_counts)) if qa_counts else 0,
        "max": max(qa_counts) if qa_counts else 0,
    }

    return {
        "passed": not errors,
        "num_images": split_image_counts,
        "num_qa": {split: qa_by_split.get(split, 0) for split in ["train", "val", "test"]},
        "qa_per_image": qa_per_image_report,
        "question_type_distribution": dict(question_type_counts),
        "answer_type_distribution": dict(answer_type_counts),
        "yes_no_ratio": yes_no_ratio,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final VQA JSONL dataset.")
    parser.add_argument("--annotations_dir", required=True, help="Directory with train/val/test.jsonl")
    parser.add_argument("--objects", required=True, help="Path to metadata/objects.jsonl")
    parser.add_argument("--processed_dir", required=True, help="Processed data root")
    parser.add_argument("--out", required=True, help="Output validation_report.json")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    annotations_dir = Path(args.annotations_dir)
    objects_path = Path(args.objects)
    processed_dir = Path(args.processed_dir)

    records = load_annotations(annotations_dir)
    object_ids = load_object_ids(objects_path)
    report = compute_report(records, object_ids, processed_dir)

    out_path = ensure_parent(args.out)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Validation report written to %s", out_path)
    logging.info("passed=%s | errors=%s | warnings=%s", report["passed"], len(report["errors"]), len(report["warnings"]))
    for error in report["errors"][:20]:
        logging.error(error)
    for warning in report["warnings"][:20]:
        logging.warning(warning)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
