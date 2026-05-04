import argparse
import csv
import json
from pathlib import Path


def truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "keep"}


def choose_answers(row):
    chosen = (row.get("corrected_chosen") or row.get("chosen") or "").strip()
    rejected = (row.get("corrected_rejected") or row.get("rejected") or "").strip()
    label = (row.get("chosen_label") or "chosen").strip().lower()

    if label in {"rejected", "b", "2"}:
        chosen, rejected = rejected, chosen
    elif label in {"tie", "ambiguous", "drop", "none"}:
        return None, None

    return chosen, rejected


def main():
    parser = argparse.ArgumentParser(
        description="Import a human-reviewed preference CSV back to JSONL."
    )
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-pairs", type=int, default=100)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    with open(args.review_csv, "r", encoding="utf-8-sig", newline="") as f_in, \
            out_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if not truthy(row.get("keep")):
                dropped += 1
                continue
            chosen, rejected = choose_answers(row)
            if not chosen or not rejected or chosen == rejected:
                dropped += 1
                continue

            item = {
                "preference_id": row.get("preference_id") or f"pref_reviewed_{kept + 1:06d}",
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "chosen": chosen,
                "rejected": rejected,
                "source": "human_reviewed_preference",
                "original_source": row.get("source"),
                "source_model": row.get("source_model"),
                "question_type": row.get("question_type"),
                "answer_type": row.get("answer_type"),
                "review_reason": row.get("reason"),
            }
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Kept reviewed pairs: {kept}")
    print(f"Dropped rows: {dropped}")
    print(f"Output: {out_path}")
    if kept < args.min_pairs:
        raise SystemExit(
            f"Not enough reviewed pairs: {kept} < required {args.min_pairs}"
        )


if __name__ == "__main__":
    main()
