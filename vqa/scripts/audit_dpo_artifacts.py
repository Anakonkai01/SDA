#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.metrics import normalize_answer


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def direction(row):
    chosen = str(row.get("chosen", "")).strip().lower()
    rejected = str(row.get("rejected", "")).strip().lower()
    if chosen == "không" and rejected == "có":
        return "khong>co"
    if chosen == "có" and rejected == "không":
        return "co>khong"
    if chosen in {"có", "không"} or rejected in {"có", "không"}:
        return "one_yesno"
    return "other"


def print_counter(title, counter, limit):
    print(f"\n{title}")
    for key, value in counter.most_common(limit):
        print(f"  {key}: {value}")


def audit_preferences(path, limit):
    rows = read_jsonl(path)
    print(f"PREFERENCES {path}")
    print(f"total={len(rows)}")
    print(f"empty={sum(not str(r.get('chosen', '')).strip() or not str(r.get('rejected', '')).strip() for r in rows)}")
    print(f"same={sum(str(r.get('chosen', '')).strip() == str(r.get('rejected', '')).strip() for r in rows)}")
    print_counter("by_question_type", Counter(r.get("question_type") or "unknown" for r in rows), limit)
    print_counter("by_answer_type", Counter(r.get("answer_type") or "unknown" for r in rows), limit)
    print_counter("directions", Counter(direction(r) for r in rows), limit)
    print_counter("top_chosen", Counter(str(r.get("chosen", "")).strip() for r in rows), limit)
    print_counter("top_rejected", Counter(str(r.get("rejected", "")).strip() for r in rows), limit)


def audit_prediction_delta(sft_path, dpo_path, limit):
    sft = read_jsonl(sft_path)
    dpo = read_jsonl(dpo_path)
    by_id = {r.get("question_id") or i: r for i, r in enumerate(sft)}
    changes = []
    buckets = defaultdict(lambda: Counter())
    for i, row in enumerate(dpo):
        key = row.get("question_id") or i
        old = by_id.get(key)
        if old is None:
            continue
        ref = normalize_answer(row.get("reference", ""))
        sft_pred = normalize_answer(old.get("prediction", ""))
        dpo_pred = normalize_answer(row.get("prediction", ""))
        if sft_pred == dpo_pred:
            continue
        qtype = row.get("question_type") or "unknown"
        if sft_pred == ref and dpo_pred != ref:
            status = "regression"
        elif sft_pred != ref and dpo_pred == ref:
            status = "fix"
        else:
            status = "changed_wrong_or_metric_close"
        buckets[qtype][status] += 1
        changes.append((status, qtype, old, row))

    print(f"\nPREDICTION DELTA sft={sft_path} dpo={dpo_path}")
    print(f"changed={len(changes)}")
    for qtype, counter in sorted(buckets.items()):
        print(qtype, dict(counter))
    print("\nExamples")
    for status, qtype, old, row in changes[:limit]:
        print(json.dumps({
            "status": status,
            "question_type": qtype,
            "question": row.get("question"),
            "reference": row.get("reference"),
            "sft": old.get("prediction"),
            "dpo": row.get("prediction"),
        }, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Audit DPO preference and prediction artifacts.")
    parser.add_argument("--preferences")
    parser.add_argument("--sft-predictions")
    parser.add_argument("--dpo-predictions")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if args.preferences:
        audit_preferences(args.preferences, args.limit)
    if args.sft_predictions and args.dpo_predictions:
        audit_prediction_delta(args.sft_predictions, args.dpo_predictions, args.limit)


if __name__ == "__main__":
    main()