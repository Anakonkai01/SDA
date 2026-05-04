"""
Run B2 eval on val set to generate predictions for DPO preference pairs.
"""
import json, sys, torch, random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.model_b import VQAModelB
from train.config import ConfigB
from tqdm import tqdm

config = ConfigB()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load B2 (v8 best checkpoint) — no LoRA in init, load manually
import torch
model_b = VQAModelB(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    lora_config=None,
    device=device,
    max_pixels=config.model.max_pixels,
    load_in_4bit=True,
    backend="qwen25",
)
# Load LoRA adapter directly (avoid double-wrap issue)
from peft import PeftModel
lora_path = "checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora"
model_b.model = PeftModel.from_pretrained(model_b.model, lora_path)
model_b.model.eval()
model_b.is_lora = True
print("B2 LoRA loaded successfully")
model_b.load_lora("checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora")
model_b.model.eval()

# Load & stratify val set
import random
random.seed(42)

val_path = "data/processed/annotations/val.jsonl"
samples = []
with open(val_path) as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

# Stratify to 4000 samples for speed (enough for ~240 wrong predictions)
from collections import defaultdict
groups = defaultdict(list)
for s in samples:
    groups[s.get("question_type", "other")].append(s)
stratified = []
keys = sorted(groups)
target = min(4000, len(samples))
while len(stratified) < target and keys:
    for k in keys:
        if groups[k] and len(stratified) < target:
            stratified.append(groups[k].pop())
samples = stratified

print(f"Val samples: {len(samples)}")

# Image base path
img_base = Path("data/processed")

# Evaluate
output_path = "predictions_b2_val.jsonl"
with open(output_path, "w", encoding="utf-8") as out_f, torch.no_grad():
    for i, sample in enumerate(tqdm(samples, desc="B2 val eval")):
        img_path = str(img_base / sample.get("image_path", ""))
        q = sample["question"]
        ref = sample["answer"]

        pred = model_b.inference(img_path, q, max_new_tokens=config.data.max_answer_length)
        row = {
            "model": "B2_Qwen25_LoRA",
            "question_id": sample.get("question_id", f"val_{i:06d}"),
            "image_id": sample.get("image_id", ""),
            "image": img_path,
            "question": q,
            "reference": ref,
            "prediction": pred,
            "question_type": sample.get("question_type", ""),
            "answer_type": sample.get("answer_type", ""),
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved {len(samples)} predictions to {output_path}")
