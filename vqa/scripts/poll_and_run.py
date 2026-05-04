#!/usr/bin/env python3
"""
Polling runner: waits for DPO training, then executes remaining steps.
Run via: nohup python scripts/poll_and_run.py &
"""
import json, os, sys, time, subprocess
from pathlib import Path

VQA_DIR = Path(__file__).parent.parent
NLP_DIR = VQA_DIR.parent / "nlp"

DPO_CHECKPOINT = VQA_DIR / "checkpoints_b2_dpo_full" / "best_lora" / "adapter_config.json"
DPO_LOG = VQA_DIR / "logs" / "dpo_full_3146.log"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def dpo_is_alive():
    try:
        out = subprocess.check_output(
            "ps aux | grep -v grep | grep train_dpo_qwen",
            shell=True, text=True, timeout=10,
        )
        return len(out.strip()) > 0
    except Exception:
        return False

def run(cmd, cwd=VQA_DIR, timeout=None):
    log(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log(f"FAILED (rc={result.returncode}): {cmd}")
        err = result.stderr[-500:] if result.stderr else "no stderr"
        log(f"stderr: {err}")
    else:
        log(f"OK")
    return result

# ── Wait for DPO ──────────────────────────────────────────────────────────────
log("Waiting for DPO training to complete...")
while not DPO_CHECKPOINT.exists():
    if not dpo_is_alive():
        log("DPO process died. Starting DPO now...")
        run(f"python train/train_dpo_qwen.py "
            f"--preferences data/preferences/b2_val_full_preferences.jsonl "
            f"--sft-lora-path checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora "
            f"--output-dir checkpoints_b2_dpo_full "
            f"--epochs 1 --batch-size 1 --learning-rate 5e-5")
    time.sleep(60)

log("DPO training complete!")

# ── Step 4: B1 full test ──────────────────────────────────────────────────────
log("Step 4/8: B1 full test eval (batch=32)...")
run(f"python evaluate/evaluate.py --model b1 --backend qwen25 --load-in-4bit "
    f"--max-pixels 501760 --eval-batch-size 32 "
    f"--output results_b1_v8_fulltest.json "
    f"--predictions-output predictions_b1_v8_fulltest.jsonl")

# ── Step 5: B2-DPO full test ──────────────────────────────────────────────────
log("Step 5/8: B2-DPO full test eval (batch=32)...")
run(f"python evaluate/evaluate.py --model b2 --backend qwen25 --load-in-4bit "
    f"--max-pixels 501760 --lora-path checkpoints_b2_dpo_full/best_lora "
    f"--eval-batch-size 32 "
    f"--output results_b2_dpo_fulltest.json "
    f"--predictions-output predictions_b2_dpo_fulltest.jsonl")

# ── Step 6: BERTScore for DPO ────────────────────────────────────────────────
log("Step 6/8: BERTScore for B2-DPO...")
run(f"python scripts/compute_bertscore_v2.py "
    f"--predictions predictions_b2_dpo_fulltest.jsonl "
    f"--output-suffix b2_dpo_fulltest")

# ── Step 7: NLP eval 4 configs ────────────────────────────────────────────────
log("Step 7/8: NLP evaluate 4 configs...")
env = os.environ.copy()
env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}"
for cmd, desc in [
    (["python", "src/evaluate.py", "--configs", "A", "B", "C", "D"], "NLP eval"),
    (["python", "src/evaluate_mc.py", "--configs", "A", "B", "C", "D"], "NLP MC eval"),
]:
    log(f"Step 8/8: {desc}...")
    subprocess.run(cmd, cwd=str(NLP_DIR), env=env, check=False)

log("=== OVERNIGHT PIPELINE COMPLETE ===")
log("Results:")
log("  VQA: results_b1_v8_fulltest.json")
log("  VQA: results_b2_dpo_fulltest.json")
log("  VQA: predictions_b2_dpo_fulltest.jsonl")
log("  NLP: reports/traffic/evaluation_results.json")
log("  NLP: reports/traffic/mc_results.json")
