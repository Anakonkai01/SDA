import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.metrics import compute_metrics, normalize_answer


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def heuristic_judge(row):
    pred = row.get("prediction", "")
    ref = row.get("reference", "")
    metrics = compute_metrics([pred], [ref])
    exact = normalize_answer(pred) == normalize_answer(ref)
    score = 1.0 if exact else max(metrics["rouge_l"], metrics["meteor"])
    if score >= 0.8:
        verdict = "correct"
    elif score >= 0.4:
        verdict = "partial"
    else:
        verdict = "incorrect"
    return {
        "score": round(float(score), 4),
        "verdict": verdict,
        "rationale": "heuristic exact/ROUGE/METEOR judge",
    }


def build_prompt(row):
    return f"""Bạn là giám khảo VQA tiếng Việt về biển báo giao thông.
Đánh giá câu trả lời dự đoán so với đáp án tham chiếu.

Quy tắc:
- score=1 nếu đúng về ý nghĩa.
- score=0.5 nếu gần đúng hoặc thiếu chi tiết nhưng vẫn hữu ích.
- score=0 nếu sai, mơ hồ, không trả lời, hoặc trái đáp án.
- Chỉ trả về JSON hợp lệ, không thêm văn bản ngoài JSON.

Question: {row.get("question")}
Reference answer: {row.get("reference")}
Predicted answer: {row.get("prediction")}

JSON schema:
{{"score": 0.0, "verdict": "correct|partial|incorrect", "rationale": "ngắn gọn"}}
"""


def parse_json_object(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def openrouter_judge(row, model):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for --provider openrouter")

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": build_prompt(row)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    parsed = parse_json_object(content)
    return {
        "score": float(parsed.get("score", 0)),
        "verdict": parsed.get("verdict", "unknown"),
        "rationale": parsed.get("rationale", ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Judge prediction JSONL with heuristic or LLM-as-a-judge."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--provider", choices=["heuristic", "openrouter"],
                        default="heuristic")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    score_sum = 0.0
    verdict_counts = {}
    with out_path.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.predictions):
            if args.limit is not None and total >= args.limit:
                break
            judge = (
                heuristic_judge(row)
                if args.provider == "heuristic"
                else openrouter_judge(row, args.model)
            )
            output = dict(row)
            output["judge_provider"] = args.provider
            output["judge_model"] = args.model if args.provider != "heuristic" else None
            output["judge_score"] = judge["score"]
            output["judge_verdict"] = judge["verdict"]
            output["judge_rationale"] = judge["rationale"]
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

            total += 1
            score_sum += judge["score"]
            verdict_counts[judge["verdict"]] = verdict_counts.get(judge["verdict"], 0) + 1

    summary = {
        "total": total,
        "mean_judge_score": score_sum / total if total else 0.0,
        "verdict_counts": verdict_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_out:
        Path(args.summary_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
