"""
Compute BERTScore for all models via direct PhoBERT, batched for speed.
Stratified 2000 samples per model.

Usage:
  python scripts/compute_bertscore_v2.py --predictions predictions_b2_dpo_full.jsonl --output-suffix dpo_full
"""
import json, random, torch, argparse
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=str, default="predictions_all_final_test.jsonl")
parser.add_argument("--output-suffix", type=str, default="")
args = parser.parse_args()

random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
STRATIFIED = 2000
PRED_FILE = args.predictions
SUFFIX = args.output_suffix

# ── Load predictions + sample ──
by_model = defaultdict(list)
with open(PRED_FILE) as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            by_model[p["model"]].append(p)

# ── Load PhoBERT ──
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
mdl = AutoModel.from_pretrained("vinai/phobert-base-v2").eval().to(device)
print(f"PhoBERT loaded on {device}")

def batch_embed(texts, batch_size=64):
    """Embed texts in batches, return list of (seq_len, 768) tensors."""
    result = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=254,
                  padding=True, add_special_tokens=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = mdl(**enc).last_hidden_state  # (B, S, 768)
        for j in range(len(batch)):
            # Strip [CLS] and [SEP] tokens
            result.append(hidden[j, 1:-1])
    return result

def compute_bertscore(preds, refs, batch_size=64):
    p_embs = batch_embed(preds, batch_size)
    r_embs = batch_embed(refs, batch_size)
    f1s = []
    for p_emb, r_emb in zip(p_embs, r_embs):
        p_norm = torch.nn.functional.normalize(p_emb, dim=-1)
        r_norm = torch.nn.functional.normalize(r_emb, dim=-1)
        sim = p_norm @ r_norm.T
        prec = sim.max(dim=1).values.mean().item()
        rec  = sim.max(dim=0).values.mean().item()
        denom = prec + rec
        f1s.append((2 * prec * rec / denom) if denom > 0 else 0.0)
    return sum(f1s) / len(f1s)

# ── Compute per model ──
results = {}
for model_name in ["A1_LSTM", "A2_Transformer", "B1_Qwen25_ZeroShot", "B2_Qwen25_LoRA"]:
    preds = by_model.get(model_name, [])
    if not preds:
        continue
    # Stratified sampling
    by_qtype = defaultdict(list)
    for p in preds:
        by_qtype[p.get("question_type", "other")].append(p)
    sampled = []
    for qtype, group in by_qtype.items():
        n = max(1, int(STRATIFIED * len(group) / len(preds)))
        sampled.extend(random.sample(group, min(n, len(group))))
    refs = [p["reference"] for p in sampled]
    gens = [p["prediction"] for p in sampled]
    print(f"\n{model_name}: {len(sampled)} samples, computing BERTScore...")
    score = compute_bertscore(gens, refs)
    print(f"  BERTScore-F1: {score:.4f}")
    results[model_name] = round(score, 4)

del mdl, tok
torch.cuda.empty_cache()

# ── Update result files ──
if SUFFIX:
    V8_FILES = {}
    for model_key in results:
        V8_FILES[model_key] = f"results_{SUFFIX}.json"
else:
    V8_FILES = {
        "A1_LSTM": "results_v8_a1_50k_fulltest.json",
        "A2_Transformer": "results_v8_a2_50k_fulltest.json",
        "B2_Qwen25_LoRA": "results_b2_qwen25_v8_50k_strat_4bit_lr5e5_fulltest.json",
    }
print(f"Updating: {list(V8_FILES.values())}")
for model_key, result_path in V8_FILES.items():
    path = Path(result_path)
    if path.exists() and model_key in results:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and model_key in data:
            data[model_key]["bertscore_f1"] = results[model_key]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"  Updated {result_path}")

meta_path = Path("results_all_final_test.json")
if meta_path.exists():
    data = json.loads(meta_path.read_text())
    for mk in list(data):
        if mk in results:
            data[mk]["bertscore_f1"] = results[mk]
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  Updated results_all_final_test.json")

print(f"\nFinal:")
for k, v in results.items():
    print(f"  {k}: {v}")
