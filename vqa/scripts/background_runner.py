#!/usr/bin/env python3
"""Background runner: poll B2-DPO eval, then BERTScore + NLP eval."""
import os, sys, time, subprocess
from pathlib import Path

VQA = Path(__file__).parent.parent
NLP = VQA.parent / "nlp"
LOG = VQA / "logs" / "background_runner.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run(cmd, cwd=VQA):
    log(f"Run: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        log(f"FAIL ({r.returncode}): {r.stderr[-200:]}")
    else:
        log("OK")

log("=== Background runner started ===")

# Wait for B2-DPO eval
log("Waiting for B2-DPO eval to finish...")
while True:
    r = subprocess.run(["systemctl", "--user", "is-active", "b2-dpo-eval.service"],
                        capture_output=True, text=True)
    if r.stdout.strip() != "active":
        break
    time.sleep(30)
log("B2-DPO eval done!")

# BERTScore
log("Running BERTScore...")
run(f"python scripts/compute_bertscore_v2.py --predictions predictions_b2_dpo_fulltest.jsonl --output-suffix b2_dpo_fulltest")

# NLP eval
log("Running NLP eval 4 configs...")
env = os.environ.copy()
env["PYTHONPATH"] = "src"
subprocess.run(["python", "src/evaluate.py", "--configs", "A", "B", "C", "D"],
               cwd=str(NLP), env=env, check=False)

log("Running NLP MC...")
subprocess.run(["python", "src/evaluate_mc.py", "--configs", "A", "B", "C", "D"],
               cwd=str(NLP), env=env, check=False)

log("=== ALL DONE ===")
