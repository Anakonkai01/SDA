import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils.dataset import load_vqa_data


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
YES_NO = {"Có", "Không"}


def normalize_answer(text):
    return " ".join(str(text or "").strip().split()).lower()


def unique_preserve_order(values):
    seen = set()
    output = []
    for value in values:
        key = normalize_answer(value)
        if key and key not in seen:
            seen.add(key)
            output.append(str(value).strip())
    return output


def load_pool_samples(data_path, pool_splits):
    samples = []
    for split in pool_splits:
        samples.extend(load_vqa_data(data_path, split))
    return samples


def build_answer_pools(samples):
    by_question_type = defaultdict(list)
    by_answer_type = defaultdict(list)
    global_answers = []

    for sample in samples:
        answer = str(sample.get("answer", "")).strip()
        if not answer:
            continue
        question_type = sample.get("question_type", sample.get("type", "unknown"))
        by_question_type[question_type].append(answer)
        by_answer_type[sample.get("answer_type", "unknown")].append(answer)
        global_answers.append(answer)

    return {
        "question_type": {
            key: unique_preserve_order(values)
            for key, values in by_question_type.items()
        },
        "answer_type": {
            key: unique_preserve_order(values)
            for key, values in by_answer_type.items()
        },
        "global": unique_preserve_order(global_answers),
    }


def numeric_distractors(correct, count):
    try:
        value = int(str(correct).strip())
    except ValueError:
        return []

    candidates = []
    for delta in (1, -1, 2, -2, 3, -3, 4):
        cand = value + delta
        if cand >= 0:
            candidates.append(str(cand))
    return candidates[:count]


def choose_distractors(sample, pools, num_distractors, rng):
    answer = str(sample.get("answer", "")).strip()
    answer_norm = normalize_answer(answer)
    question_type = sample.get("question_type", sample.get("type", "unknown"))
    answer_type = sample.get("answer_type", "unknown")

    if answer_type == "yes_no" or answer in YES_NO:
        return [value for value in ("Có", "Không") if normalize_answer(value) != answer_norm]

    if answer_type == "number":
        candidates = []
        candidates.extend(numeric_distractors(answer, num_distractors + 2))
        candidates.extend(pools["answer_type"].get(answer_type, []))
        candidates = [
            value for value in unique_preserve_order(candidates)
            if normalize_answer(value) != answer_norm
        ]
        rng.shuffle(candidates)
        return candidates[:num_distractors]

    candidates = list(pools["question_type"].get(question_type, []))
    candidates = [
        value for value in unique_preserve_order(candidates)
        if normalize_answer(value) != answer_norm
    ]
    if len(candidates) < num_distractors:
        candidates.extend(pools["answer_type"].get(answer_type, []))
        candidates = [
            value for value in unique_preserve_order(candidates)
            if normalize_answer(value) != answer_norm
        ]
    if len(candidates) < num_distractors:
        candidates.extend(
            value for value in pools["global"]
            if normalize_answer(value) != answer_norm
        )
        candidates = unique_preserve_order(candidates)
    rng.shuffle(candidates)
    return candidates[:num_distractors]


def build_mc_rows(samples, pools, choices, seed):
    rng = random.Random(seed)
    rows = []
    qtype_counts = Counter()

    for sample in samples:
        answer = str(sample.get("answer", "")).strip()
        if not answer:
            continue

        target_choices = 2 if sample.get("answer_type") == "yes_no" else choices
        distractors = choose_distractors(sample, pools, target_choices - 1, rng)
        choice_texts = unique_preserve_order([answer] + distractors)

        if len(choice_texts) < 2:
            continue

        rng.shuffle(choice_texts)
        correct_index = next(
            idx for idx, text in enumerate(choice_texts)
            if normalize_answer(text) == normalize_answer(answer)
        )

        choices_payload = [
            {"label": LETTERS[idx], "text": text}
            for idx, text in enumerate(choice_texts)
        ]

        row = {
            "question_id": sample.get("question_id"),
            "image_id": sample.get("image_id"),
            "image": sample.get("image"),
            "question": sample.get("question"),
            "answer": answer,
            "question_type": sample.get("question_type", sample.get("type")),
            "answer_type": sample.get("answer_type"),
            "choices": choices_payload,
            "correct_index": correct_index,
            "correct_label": LETTERS[correct_index],
            "split": sample.get("split"),
            "source": sample.get("source"),
        }
        qtype_counts[row["question_type"]] += 1
        rows.append(row)

    return rows, qtype_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/annotations")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--pool-splits",
        default="train,val,test",
        help="Comma-separated splits used to build distractor answer space.",
    )
    parser.add_argument("--choices", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pool_splits = [split.strip() for split in args.pool_splits.split(",") if split.strip()]
    pool_samples = load_pool_samples(args.data, pool_splits)
    eval_samples = load_vqa_data(args.data, args.split)
    pools = build_answer_pools(pool_samples)
    rows, qtype_counts = build_mc_rows(
        eval_samples,
        pools,
        choices=args.choices,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"MC rows saved: {output_path}")
    print(f"Rows: {len(rows)}")
    print("Question types:")
    for key, count in qtype_counts.most_common():
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
