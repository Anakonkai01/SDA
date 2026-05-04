import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "preference_id",
    "keep",
    "chosen_label",
    "image",
    "image_id",
    "question_id",
    "question",
    "reference",
    "chosen",
    "rejected",
    "corrected_chosen",
    "corrected_rejected",
    "reason",
    "source",
    "source_model",
    "question_type",
    "answer_type",
]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="Export preference JSONL to a human-review CSV."
    )
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in read_jsonl(args.preferences):
            out = {key: "" for key in FIELDNAMES}
            out.update({
                "preference_id": row.get("preference_id"),
                "keep": "1",
                "chosen_label": "chosen",
                "image": row.get("image"),
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "reference": row.get("chosen"),
                "chosen": row.get("chosen"),
                "rejected": row.get("rejected"),
                "source": row.get("source"),
                "source_model": row.get("source_model"),
                "question_type": row.get("question_type"),
                "answer_type": row.get("answer_type"),
            })
            writer.writerow(out)
            count += 1
            if args.max_rows is not None and count >= args.max_rows:
                break

    print(f"Wrote review rows: {count}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
