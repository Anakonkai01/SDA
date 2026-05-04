import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            row = dict(row)
            row["preference_id"] = f"pref_{i:06d}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value):
    return str(value or "").strip().lower()


def pair_direction(row):
    chosen = normalize_text(row.get("chosen"))
    rejected = normalize_text(row.get("rejected"))
    if chosen == "không" and rejected == "có":
        return "khong_gt_co"
    if chosen == "có" and rejected == "không":
        return "co_gt_khong"
    return "other"


def balanced_select(rows, max_total, max_per_type, max_per_chosen, max_khong_gt_co):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("question_type") or "unknown"].append(row)

    selected = []
    chosen_counts = Counter()
    type_counts = Counter()
    direction_counts = Counter()
    keys = sorted(groups)

    # Round-robin by question type so one frequent failure mode cannot dominate.
    while keys and len(selected) < max_total:
        next_keys = []
        added_this_round = False
        for key in keys:
            bucket = groups[key]
            while bucket:
                row = bucket.pop(0)
                chosen_key = normalize_text(row.get("chosen"))
                direction = pair_direction(row)
                if type_counts[key] >= max_per_type:
                    continue
                if chosen_counts[chosen_key] >= max_per_chosen:
                    continue
                if direction == "khong_gt_co" and direction_counts[direction] >= max_khong_gt_co:
                    continue
                selected.append(row)
                type_counts[key] += 1
                chosen_counts[chosen_key] += 1
                direction_counts[direction] += 1
                added_this_round = True
                break
            if bucket and type_counts[key] < max_per_type:
                next_keys.append(key)
            if len(selected) >= max_total:
                break
        if not added_this_round:
            break
        keys = next_keys
    return selected


def print_audit(rows, title):
    print(f"\n{title}")
    print(f"total={len(rows)}")
    print("by_question_type", Counter(r.get("question_type") or "unknown" for r in rows).most_common())
    print("top_chosen", Counter(r.get("chosen") for r in rows).most_common(20))
    print("top_rejected", Counter(r.get("rejected") for r in rows).most_common(20))
    print("directions", Counter(pair_direction(r) for r in rows).most_common())


def main():
    parser = argparse.ArgumentParser(description="Build balanced DPO preference pairs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-total", type=int, default=1000)
    parser.add_argument("--max-per-type", type=int, default=150)
    parser.add_argument("--max-per-chosen", type=int, default=100)
    parser.add_argument("--max-khong-gt-co", type=int, default=120)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    print_audit(rows, "INPUT AUDIT")
    selected = balanced_select(
        rows,
        max_total=args.max_total,
        max_per_type=args.max_per_type,
        max_per_chosen=args.max_per_chosen,
        max_khong_gt_co=args.max_khong_gt_co,
    )
    print_audit(selected, "OUTPUT AUDIT")
    write_jsonl(args.out, selected)
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()