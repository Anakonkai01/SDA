#!/usr/bin/env python3
"""Chain runner v3: B2-DPO eval → BERTScore → NLP eval → NLP MC"""
import os, sys, time, subprocess
from pathlib import Path

VQA = Path(__file__).parent.parent
NLP = VQA.parent / "nlp"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run(cmd, cwd=VQA):
    log(f"Run: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode: log(f"FAIL: {r.stderr[-200:]}")
    else: log("OK")

log("Waiting for B2-DPO eval...")
while subprocess.run(["systemctl","--user","is-active","b2-dpo-eval.service"],
                     capture_output=True).stdout.strip() == "active":
    time.sleep(30)
log("B2-DPO done!")

# BERTScore
log("BERTScore...")
run(f"python scripts/compute_bertscore_v2.py --predictions predictions_b2_dpo_fulltest.jsonl --output-suffix b2_dpo_fulltest")

# NLP eval with new fine-tuned model
log("NLP eval...")
env = os.environ.copy()
env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH','')}"
subprocess.run(["python","src/evaluate.py","--configs","A","B","C","D"],
               cwd=str(NLP), env=env, check=False)
log("NLP MC...")
subprocess.run(["python","src/evaluate_mc.py","--configs","A","B","C","D"],
               cwd=str(NLP), env=env, check=False)
log("=== ALL DONE ===")
