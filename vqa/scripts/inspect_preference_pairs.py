import argparse
import json
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Inspect preference JSONL stats.")
    parser.add_argument("--preferences", required=True)
    args = parser.parse_args()

    total = 0
    source_models = Counter()
    question_types = Counter()
    empty = 0
    same = 0

    with open(args.preferences, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            source_models[row.get("source_model") or "unknown"] += 1
            question_types[row.get("question_type") or "unknown"] += 1
            chosen = str(row.get("chosen") or "").strip()
            rejected = str(row.get("rejected") or "").strip()
            if not chosen or not rejected:
                empty += 1
            if chosen == rejected:
                same += 1

    print(f"total={total}")
    print(f"empty_chosen_or_rejected={empty}")
    print(f"same_chosen_rejected={same}")
    print(f"source_models={dict(source_models.most_common())}")
    print(f"question_types={dict(question_types.most_common())}")


if __name__ == "__main__":
    main()
