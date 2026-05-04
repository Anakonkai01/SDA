"""
Compute BERTScore (PhoBERT) for all VQA models from existing predictions.
Runs on GPU alongside NLP fine-tune (PhoBERT ~1.5GB).
"""
import json, gc, torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

PRED_FILE = "predictions_all_final_test.jsonl"
META_FILE = "results_all_final_test.json"
V8_RESULTS = {
    "A1": "results_v8_a1_50k_fulltest.json",
    "A2": "results_v8_a2_50k_fulltest.json",
    "B2": "results_b2_qwen25_v8_50k_strat_4bit_lr5e5_fulltest.json",
}

# ── Load predictions ──
by_model = defaultdict(list)
with open(PRED_FILE) as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            by_model[p["model"]].append(p)

print(f"Models found: {list(by_model.keys())}")
for m, preds in by_model.items():
    print(f"  {m}: {len(preds)} predictions")

# ── PhoBERT BERTScore (same logic as evaluate.py compute_bert_score) ──
def load_phobert():
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = AutoModel.from_pretrained("vinai/phobert-base-v2").eval().to(device)
    print(f"  PhoBERT on {device}")
    return tok, mdl

def compute_bert_score(predictions, references, tok, mdl):
    device = next(mdl.parameters()).device
    def embed(texts):
        result = []
        for text in texts:
            enc = tok(text, return_tensors="pt", truncation=True, max_length=254, padding=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                hidden = mdl(**enc).last_hidden_state[0]
            result.append(hidden[1:-1])
        return result

    pred_embs = embed(predictions)
    ref_embs  = embed(references)
    f1_scores = []
    for p_emb, r_emb in zip(pred_embs, ref_embs):
        p_norm = torch.nn.functional.normalize(p_emb, dim=-1)
        r_norm = torch.nn.functional.normalize(r_emb, dim=-1)
        sim = p_norm @ r_norm.T
        prec = sim.max(dim=1).values.mean().item()
        rec  = sim.max(dim=0).values.mean().item()
        denom = prec + rec
        f1_scores.append((2 * prec * rec / denom) if denom > 0 else 0.0)
    return sum(f1_scores) / len(f1_scores)

# ── Compute BERTScore per model ──
print("\nLoading PhoBERT...")
tok, mdl = load_phobert()

results = {}
for model_name in ["A1_LSTM", "A2_Transformer", "B1_Qwen25_ZeroShot", "B2_Qwen25_LoRA"]:
    preds = by_model.get(model_name, [])
    if not preds:
        print(f"  {model_name}: no predictions, skipping")
        continue
    refs  = [p["reference"] for p in preds]
    gens  = [p["prediction"] for p in preds]
    print(f"\n{model_name}: computing BERTScore on {len(gens)} samples...")
    score = compute_bert_score(gens, refs, tok, mdl)
    print(f"  BERTScore-F1: {score:.4f}")
    results[model_name] = round(score, 4)

del mdl, tok
gc.collect()
torch.cuda.empty_cache()

# ── Update result files ──
for config_key, result_path in [("A1_LSTM", "results_v8_a1_50k_fulltest.json"),
                                  ("A2_Transformer", "results_v8_a2_50k_fulltest.json"),
                                  ("B2_Qwen25_LoRA", "results_b2_qwen25_v8_50k_strat_4bit_lr5e5_fulltest.json")]:
    path = Path(result_path)
    if path.exists():
        data = json.loads(path.read_text())
        for model_key, model_data in data.items() if isinstance(data, dict) else [(config_key, data)]:
            if model_key == config_key or len(data) == 1:
                model_data["bertscore_f1"] = results.get(config_key)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"  Updated {result_path}")

# Also update v5 results file
meta_path = Path(META_FILE)
if meta_path.exists():
    data = json.loads(meta_path.read_text())
    for model_key in list(data):
        if model_key in results:
            data[model_key]["bertscore_f1"] = results[model_key]
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  Updated {META_FILE}")

print(f"\nFinal BERTScore results:")
for k, v in results.items():
    print(f"  {k}: {v}")
