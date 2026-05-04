import argparse
import csv
import html
import json
import os
import re
from collections import Counter


def normalize_answer(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.。!?！？]+$", "", text).strip()
    prefixes = [
        "biển báo này là ",
        "đây là ",
        "đáp án là ",
        "câu trả lời là ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def read_rows(path, model=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if model and row.get("model") != model:
                continue
            ref_norm = normalize_answer(row.get("reference"))
            pred_norm = normalize_answer(row.get("prediction"))
            row["reference_norm"] = ref_norm
            row["prediction_norm"] = pred_norm
            row["exact_match"] = str(row.get("reference", "")).strip() == str(
                row.get("prediction", "")
            ).strip()
            row["normalized_match"] = ref_norm == pred_norm
            rows.append(row)
    return rows


def filter_rows(rows, mode, question_type=None, answer_type=None):
    out = []
    for row in rows:
        if mode == "wrong" and row["exact_match"]:
            continue
        if mode == "normalized_wrong" and row["normalized_match"]:
            continue
        if mode == "rescued" and (row["exact_match"] or not row["normalized_match"]):
            continue
        if question_type and row.get("question_type") != question_type:
            continue
        if answer_type and row.get("answer_type") != answer_type:
            continue
        out.append(row)
    return out


def write_csv(rows, output_path):
    fields = [
        "model",
        "question_id",
        "image_id",
        "image",
        "question_type",
        "answer_type",
        "question",
        "reference",
        "prediction",
        "reference_norm",
        "prediction_norm",
        "exact_match",
        "normalized_match",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_html(rows, output_path, repo_root):
    cards = []
    for row in rows:
        image_path = row.get("image", "")
        abs_image = os.path.abspath(os.path.join(repo_root, image_path))
        cells = {
            "Question": row.get("question", ""),
            "Reference": row.get("reference", ""),
            "Prediction": row.get("prediction", ""),
            "Question type": row.get("question_type", ""),
            "Answer type": row.get("answer_type", ""),
            "Exact": row.get("exact_match"),
            "Normalized": row.get("normalized_match"),
            "Image": image_path,
        }
        details = "\n".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in cells.items()
        )
        cards.append(
            f"""
            <article class="card">
              <img src="file://{html.escape(abs_image)}" loading="lazy" />
              <table>{details}</table>
            </article>
            """
        )

    page = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>Prediction Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f6f4ef; color: #1d1b16; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #ddd3c1; border-radius: 14px; padding: 12px; }}
    img {{ width: 100%; max-height: 260px; object-fit: contain; background: #eee; border-radius: 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th {{ text-align: left; width: 120px; color: #6c5f4a; vertical-align: top; }}
    td, th {{ border-top: 1px solid #eee5d6; padding: 6px; }}
  </style>
</head>
<body>
  <h1>Prediction Review</h1>
  <p>Total rows: {len(rows)}</p>
  <section class="grid">{''.join(cards)}</section>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(page)


def print_summary(rows):
    total = len(rows)
    exact = sum(row["exact_match"] for row in rows)
    norm = sum(row["normalized_match"] for row in rows)
    print(f"Rows: {total}")
    if total:
        print(f"Exact match: {exact}/{total} = {exact / total:.4f}")
        print(f"Normalized match: {norm}/{total} = {norm / total:.4f}")
    print("Wrong by question_type:")
    q_counter = Counter(
        row.get("question_type")
        for row in rows
        if not row["exact_match"]
    )
    for key, count in q_counter.most_common(12):
        print(f"  {key}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model", default="B2_Qwen25_LoRA")
    parser.add_argument(
        "--mode",
        choices=["all", "wrong", "normalized_wrong", "rescued"],
        default="wrong",
    )
    parser.add_argument("--question-type", default=None)
    parser.add_argument("--answer-type", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--html-output", default=None)
    args = parser.parse_args()

    repo_root = os.getcwd()
    rows = read_rows(args.predictions, model=args.model)
    print_summary(rows)
    rows = filter_rows(
        rows,
        args.mode,
        question_type=args.question_type,
        answer_type=args.answer_type,
    )
    if args.limit is not None and args.limit > 0:
        rows = rows[:args.limit]

    if args.csv_output:
        write_csv(rows, args.csv_output)
        print(f"CSV saved: {args.csv_output}")
    if args.html_output:
        write_html(rows, args.html_output, repo_root)
        print(f"HTML saved: {args.html_output}")

    if not args.csv_output and not args.html_output:
        for row in rows[:20]:
            print("\nQ:", row.get("question"))
            print("REF:", row.get("reference"))
            print("PRED:", row.get("prediction"))
            print("IMAGE:", row.get("image"))


if __name__ == "__main__":
    main()
