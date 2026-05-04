#!/usr/bin/env python3
"""
One-shot script: DPO train (checkpoint every 250 pairs) → eval final → shutdown.

Usage: python scripts/dpo_train_eval_shutdown.py
"""
import os, sys, time, subprocess, json, shutil
from pathlib import Path

VQA = Path(__file__).parent.parent
PY = "/home/pc5070ti/miniforge3/envs/ai/bin/python"  # conda env python

def log(msg):
    t = time.strftime('%H:%M:%S')
    print(f"[{t}] {msg}", flush=True)

def run(cmd, cwd=VQA, timeout=None):
    full_cmd = f"{PY} {cmd}" if not cmd.startswith(PY) else cmd
    log(f"Run: {cmd[:100]}")
    try:
        r = subprocess.run(full_cmd, shell=True, cwd=cwd, timeout=timeout)
        if r.returncode != 0:
            log(f"FAILED (rc={r.returncode})")
        else:
            log("OK")
        return r
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT (>{timeout}s) — continuing...")
        return None

# ── Step 1: DPO Training ──────────────────────────────────────────────────────
log("=== STEP 1: DPO Training (3.5h timeout) ===")
r = run(f"-u train/train_dpo_qwen.py "
    f"--preferences data/preferences/b2_val_full_preferences.jsonl "
    f"--sft-lora-path checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora "
    f"--output-dir checkpoints_b2_dpo_full "
    f"--epochs 1 --batch-size 1 --learning-rate 1e-6 "
    f"--max-samples 3146 --checkpoint-every 250",
    timeout=14400)

# ── Step 2: Test each checkpoint to find the best one ──────────────────────────
log("=== STEP 2: Find best checkpoint ===")
checkpoint_dir = VQA / "checkpoints_b2_dpo_full"
checkpoints = sorted(checkpoint_dir.glob("checkpoint_*pairs"))
log(f"Found {len(checkpoints)} periodic checkpoints")

best_acc = 0.0
best_checkpoint = None

for ckpt in checkpoints:
    pairs = ckpt.name
    log(f"  Testing {pairs}...")
    r = subprocess.run(
        f"{PY} evaluate/evaluate.py --model b2 --backend qwen25 --load-in-4bit "
        f"--max-pixels 501760 --lora-path {ckpt.resolve()} "
        f"--stratified-limit 200 --eval-batch-size 32 "
        f"--output /tmp/dpo_ckpt_test.json",
        shell=True, cwd=VQA, capture_output=True, text=True,
    )
    if r.returncode == 0:
        try:
            # Parse accuracy from evaluate.py stdout
            for line in r.stdout.split('\n'):
                if 'vqa_accuracy' in line:
                    acc_str = line.split(':')[1].split(',')[0].strip()
                    acc = float(acc_str)
                    log(f"    Acc={acc:.4f}")
                    if acc > best_acc:
                        best_acc = acc
                        best_checkpoint = ckpt
                    break
        except Exception as e:
            log(f"    Parse error: {e}")
    else:
        log(f"    FAILED or degraded (mode collapse, rc={r.returncode})")

if best_checkpoint:
    final = checkpoint_dir / "best_lora"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(str(best_checkpoint), str(final))
    log(f"Best: {best_checkpoint.name} (acc={best_acc:.4f}) → {final}")
else:
    log("No valid checkpoint found — all degraded")

# ── Step 3: Full test eval of best checkpoint ──────────────────────────────────
log("=== STEP 3: Full test eval ===")
if best_checkpoint:
    run(f"evaluate/evaluate.py --model b2 --backend qwen25 --load-in-4bit "
        f"--max-pixels 501760 --lora-path {final.resolve()} "
        f"--eval-batch-size 32 "
        f"--output results_b2_dpo_fulltest.json "
        f"--predictions-output predictions_b2_dpo_fulltest.jsonl")
else:
    final = checkpoint_dir / "last_lora"
    if final.exists():
        run(f"evaluate/evaluate.py --model b2 --backend qwen25 --load-in-4bit "
            f"--max-pixels 501760 --lora-path {final.resolve()} "
            f"--eval-batch-size 32 "
            f"--output results_b2_dpo_fulltest.json "
            f"--predictions-output predictions_b2_dpo_fulltest.jsonl")
    else:
        log("No checkpoint at all — skipping eval")

# ── Step 4: Shutdown ────────────────────────────────────────────────────────
log("=== STEP 4: Shutdown in 30s ===")
time.sleep(30)
try:
    subprocess.run(["systemctl", "poweroff"], check=True, timeout=10)
except Exception as e:
    log(f"Cannot shutdown ({e}). Please shutdown manually.")
