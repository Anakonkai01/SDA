#!/usr/bin/env bash
# Over-night pipeline: runs remaining steps after DPO training completes.
# 
# Usage: bash scripts/run_overnight.sh
# 
# Steps:
#   1. Wait for DPO training (checkpoints_b2_dpo_full/best_lora)
#   2. B1 full test eval
#   3. B2-DPO full test eval
#   4. BERTScore for DPO
#   5. NLP eval 4 configs
#   6. NLP MC eval
#
# All times are estimates with batch=32.

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

# ── Step 0: Wait for DPO training to finish ──────────────────────────────────
log "Waiting for DPO training to complete..."
DPO_DIR="checkpoints_b2_dpo_full"
DPO_DONE="$DPO_DIR/best_lora/adapter_config.json"
while [ ! -f "$DPO_DONE" ]; do
    if ps aux | grep -q "[t]rain_dpo_qwen"; then
        sleep 60
    else
        log "DPO process not found. Checking if checkpoint exists..."
        if [ -f "$DPO_DONE" ]; then
            log "DPO checkpoint found!"
            break
        else
            log "DPO not running and no checkpoint. Starting DPO..."
            python train/train_dpo_qwen.py \
                --preferences data/preferences/b2_val_full_preferences.jsonl \
                --sft-lora-path checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora \
                --output-dir "$DPO_DIR" \
                --epochs 1 --batch-size 1 --learning-rate 5e-5 \
                > logs/dpo_overnight.log 2>&1
        fi
    fi
done
log "DPO training complete."

# ── Step 4: B1 full test eval (batch=32) ──────────────────────────────────────
log "Step 4: B1 full test eval..."
python evaluate/evaluate.py \
    --model b1 --backend qwen25 --load-in-4bit --max-pixels 501760 \
    --eval-batch-size 32 \
    --output results_b1_v8_fulltest.json \
    --predictions-output predictions_b1_v8_fulltest.jsonl
log "B1 full test done."

# ── Step 5: B2-DPO full test eval (batch=32) ──────────────────────────────────
log "Step 5: B2-DPO full test eval..."
python evaluate/evaluate.py \
    --model b2 --backend qwen25 --load-in-4bit --max-pixels 501760 \
    --lora-path "$DPO_DIR/best_lora" \
    --eval-batch-size 32 \
    --output results_b2_dpo_fulltest.json \
    --predictions-output predictions_b2_dpo_fulltest.jsonl
log "B2-DPO full test done."

# ── Step 6: BERTScore for DPO ────────────────────────────────────────────────
log "Step 6: BERTScore for B2-DPO..."
python scripts/compute_bertscore_v2.py \
    --predictions predictions_b2_dpo_fulltest.jsonl \
    --output-suffix b2_dpo_fulltest
log "BERTScore done."

# ── Step 7: NLP eval 4 configs ────────────────────────────────────────────────
log "Step 7: NLP evaluate 4 configs..."
cd "$PROJECT_ROOT/../nlp"
PYTHONPATH=src:$PYTHONPATH python src/evaluate.py --configs A B C D
log "NLP eval done."

# ── Step 8: NLP MC eval ──────────────────────────────────────────────────────
log "Step 8: NLP MC evaluate..."
PYTHONPATH=src:$PYTHONPATH python src/evaluate_mc.py --configs A B C D
log "NLP MC eval done."

cd "$PROJECT_ROOT"

log "=== OVERNIGHT PIPELINE COMPLETE ==="
log "Results:"
log "  results_b1_v8_fulltest.json"
log "  results_b2_dpo_fulltest.json"
log "  predictions_b2_dpo_fulltest.jsonl"
log "  nlp/reports/traffic/evaluation_results.json"
log "  nlp/reports/traffic/mc_results.json"
