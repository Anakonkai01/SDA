#!/usr/bin/env python3
"""
Chain runner: B1 eval → B2-DPO eval → BERTScore → NLP eval → NLP MC
"""
import os, sys, time, subprocess, json
from pathlib import Path

VQA_DIR = Path(__file__).parent.parent
NLP_DIR = VQA_DIR.parent / "nlp"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run(cmd, cwd=VQA_DIR):
    log(f"Running: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"FAILED: {r.stderr[-300:]}")
    else:
        log(f"OK")
    return r

# Step 1: Wait for B1 eval
log("Waiting for B1 eval to complete...")
while subprocess.run(["systemctl", "--user", "is-active", "b1-v8.service"],
                      capture_output=True).stdout.strip() == "active":
    time.sleep(30)

# Step 2: B2-DPO full test
log("B2-DPO full test...")
run(f"python evaluate/evaluate.py --model b2 --backend qwen25 --load-in-4bit "
    f"--max-pixels 501760 --lora-path checkpoints_b2_dpo_full/best_lora "
    f"--eval-batch-size 32 --output results_b2_dpo_fulltest.json "
    f"--predictions-output predictions_b2_dpo_fulltest.jsonl")

# Step 3: BERTScore for DPO
log("BERTScore...")
run(f"python scripts/compute_bertscore_v2.py "
    f"--predictions predictions_b2_dpo_fulltest.jsonl "
    f"--output-suffix b2_dpo_fulltest")

# Step 4: NLP eval
log("NLP eval 4 configs...")
env = os.environ.copy()
env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}"
for cmd, name in [
    (["python", "src/evaluate.py", "--configs", "A", "B", "C", "D"], "NLP eval"),
    (["python", "src/evaluate_mc.py", "--configs", "A", "B", "C", "D"], "NLP MC"),
]:
    log(f"{name}...")
    subprocess.run(cmd, cwd=str(NLP_DIR), env=env, check=False)

log("=== ALL DONE ===")
