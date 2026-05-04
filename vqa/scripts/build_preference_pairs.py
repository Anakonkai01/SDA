import argparse
import json
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


def is_bad_prediction(prediction, reference):
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred:
        return True
    if pred == ref:
        return False
    vague = {
        "dont know",
        "don t know",
        "unknown",
        "không biết",
        "khong biet",
    }
    return pred in vague or pred != ref


def build_pairs(rows, max_pairs=None):
    pairs = []
    for row in rows:
        reference = row.get("reference")
        prediction = row.get("prediction")
        if reference is None or prediction is None:
            continue
        if not is_bad_prediction(prediction, reference):
            continue

        pref_id = f"pref_{len(pairs) + 1:06d}"
        pairs.append({
            "preference_id": pref_id,
            "image": row.get("image"),
            "image_id": row.get("image_id"),
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "chosen": reference,
            "rejected": prediction,
            "source": "gold_vs_model_prediction",
            "source_model": row.get("model"),
            "question_type": row.get("question_type"),
            "answer_type": row.get("answer_type"),
        })
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Build preference pairs from evaluate.py prediction JSONL."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    pairs = build_pairs(rows, args.max_pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Read predictions: {len(rows)}")
    print(f"Wrote preference pairs: {len(pairs)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
