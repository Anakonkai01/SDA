"""
LLM-as-a-judge via OpenRouter: google/gemma-3-27b-it:free
Chấm điểm predictions của 4 configs A1/A2/B1/B2 (200 stratified samples mỗi model).
"""
import json, os, re, random, time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("../nlp/.env")

random.seed(42)
PRED_FILE = "predictions_all_final_test.jsonl"
JUDGE_MODEL = "google/gemma-4-31b-it:free"
SAMPLES_PER_MODEL = 100

JUDGE_SYSTEM = (
    "Bạn là chuyên gia đánh giá hệ thống hỏi đáp về biển báo giao thông Việt Nam. "
    "Nhiệm vụ của bạn là chấm điểm câu trả lời của mô hình AI so với đáp án chuẩn. "
    "Chỉ trả lời bằng đúng một chữ số từ 1 đến 5, không giải thích thêm."
)

JUDGE_TEMPLATE = """Câu hỏi: {question}

Đáp án chuẩn: {reference}

Câu trả lời của mô hình: {prediction}

Thang điểm:
5 — Hoàn toàn chính xác và đầy đủ
4 — Phần lớn chính xác, thiếu vài chi tiết nhỏ
3 — Đúng về đại thể nhưng thiếu thông tin quan trọng
2 — Có phần đúng nhưng sai nhiều điểm
1 — Sai hoàn toàn hoặc không liên quan

Điểm (1-5):"""

client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY", ""), base_url="https://openrouter.ai/api/v1")

def judge_one(question, reference, prediction):
    prompt = JUDGE_TEMPLATE.format(question=question, reference=reference, prediction=prediction)
    full_prompt = f"{JUDGE_SYSTEM}\n\n{prompt}"
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.0,
            max_tokens=8,
            extra_body={"provider": {"allow_fallbacks": True}},
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"[1-5]", text)
        return int(m.group()) if m else None
    except Exception as e:
        print(f"    API error: {e}")
        return None

# ── Load + stratify ──
by_model = defaultdict(list)
with open(PRED_FILE) as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            by_model[p["model"]].append(p)

results = {}
for model_name in ["A1_LSTM", "A2_Transformer", "B1_Qwen25_ZeroShot", "B2_Qwen25_LoRA"]:
    preds = by_model.get(model_name, [])
    if not preds:
        continue
    by_qtype = defaultdict(list)
    for p in preds:
        by_qtype[p.get("question_type", "other")].append(p)
    sampled = []
    for qtype, group in by_qtype.items():
        n = max(1, int(SAMPLES_PER_MODEL * len(group) / len(preds)))
        sampled.extend(random.sample(group, min(n, len(group))))

    scores = []
    failed = 0
    print(f"\n{model_name}: judging {len(sampled)} samples via {JUDGE_MODEL}...")
    for i, p in enumerate(sampled):
        score = judge_one(p["question"], p["reference"], p["prediction"])
        if score:
            scores.append(score)
        else:
            failed += 1
        time.sleep(3)  # rate limit: 20 req/min
        if score:
            scores.append(score)
        else:
            failed += 1
        if (i + 1) % 40 == 0:
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {i+1}/{len(sampled)} — avg={avg:.3f}/5 ({avg/5:.3f} norm), failed={failed}")

    if scores:
        mean = sum(scores) / len(scores)
        results[model_name] = {"raw": round(mean, 3), "norm": round(mean / 5, 4), "n": len(scores), "failed": failed}

print(f"\n{'='*60}")
print(f"LLM-as-Judge results ({JUDGE_MODEL})")
print(f"{'='*60}")
for k, v in sorted(results.items(), key=lambda x: -x[1]["norm"]):
    print(f"  {k:<25} {v['raw']}/5  (norm={v['norm']})  [n={v['n']}, failed={v['failed']}]")

# Save results
out = {"judge_model": JUDGE_MODEL, "results": results}
Path("llm_judge_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\nSaved: llm_judge_results.json")
