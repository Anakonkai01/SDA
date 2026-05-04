import argparse
import logging
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io_utils import read_csv, write_csv


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set keep=0 for rows listed in a priority queue so the dataset can move forward."
    )
    parser.add_argument("--review_csv", required=True, help="Full input review CSV")
    parser.add_argument("--priority_csv", required=True, help="Priority queue CSV containing question_id and priority")
    parser.add_argument("--out_csv", required=True, help="Output review CSV ready for build_final_jsonl")
    parser.add_argument("--drop_priorities", nargs="+", default=["high"], help="Priorities to drop, default: high")
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    rows = read_csv(args.review_csv)
    priority_rows = read_csv(args.priority_csv)
    drop_priorities = {value.strip().lower() for value in args.drop_priorities}
    drop_ids = {
        str(row.get("question_id", "")).strip()
        for row in priority_rows
        if str(row.get("priority", "")).strip().lower() in drop_priorities
    }

    dropped = 0
    for row in rows:
        if str(row.get("question_id", "")).strip() in drop_ids:
            row["keep"] = "0"
            dropped += 1

    write_csv(args.out_csv, rows, list(rows[0].keys()) if rows else [])
    logging.info("Wrote %s rows to %s", len(rows), args.out_csv)
    logging.info("Dropped rows=%s for priorities=%s", dropped, sorted(drop_priorities))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
