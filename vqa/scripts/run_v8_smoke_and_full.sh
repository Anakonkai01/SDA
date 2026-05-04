#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/incoming/extracted/processed_archive_all_stratified_801010/annotations_rulebased_archive_all_stratified_v8_1to4_4q_per_type}"
SEED="${SEED:-42}"
LOG_DIR="${LOG_DIR:-logs/v8_$(date +%Y%m%d_%H%M%S)}"

SMOKE_TRAIN="${SMOKE_TRAIN:-96}"
SMOKE_VAL="${SMOKE_VAL:-48}"
SMOKE_TEST="${SMOKE_TEST:-96}"

A_SAMPLES="${A_SAMPLES:-50000}"
A_VAL_SAMPLES="${A_VAL_SAMPLES:-2000}"
A_EPOCHS="${A_EPOCHS:-5}"
A_BATCH_SIZE="${A_BATCH_SIZE:-32}"

B_SAMPLES="${B_SAMPLES:-50000}"
B_VAL_SAMPLES="${B_VAL_SAMPLES:-1000}"
B_EPOCHS="${B_EPOCHS:-1}"
B_BATCH_SIZE="${B_BATCH_SIZE:-1}"
B_GRAD_ACCUM="${B_GRAD_ACCUM:-4}"
B_LR="${B_LR:-5e-5}"
EVAL_STRATIFIED="${EVAL_STRATIFIED:-0}"
USE_WANDB="${USE_WANDB:-1}"

WANDB_ARGS=()
if [[ "$USE_WANDB" != "1" ]]; then
  WANDB_ARGS=(--no-wandb)
fi

if [[ "$EVAL_STRATIFIED" == "1" ]]; then
  EVAL_ARGS=(--stratified-limit 1000 --stratified-seed "$SEED")
  EVAL_SUFFIX="strat1000"
else
  EVAL_ARGS=()
  EVAL_SUFFIX="fulltest"
fi

mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG_DIR/run.log"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] DONE  $name" | tee -a "$LOG_DIR/run.log"
}

echo "Logs: $LOG_DIR" | tee "$LOG_DIR/run.log"
echo "Data: $DATA_DIR" | tee -a "$LOG_DIR/run.log"

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  run_step smoke_a1 \
    python train/train_a.py \
      --data "$DATA_DIR" \
      --decoder lstm \
      --batch-size 8 \
      --phase1-epochs 1 \
      --phase2-epochs 0 \
      --stratified-train-samples "$SMOKE_TRAIN" \
      --stratified-val-samples "$SMOKE_VAL" \
      --sample-seed "$SEED" \
      --checkpoint-dir checkpoints_v8_smoke \
      "${WANDB_ARGS[@]}"

  run_step smoke_a1_eval \
    python evaluate/evaluate.py \
      --data "$DATA_DIR" \
      --model a1 \
      --checkpoint checkpoints_v8_smoke/model_a1/best.pt \
      --stratified-limit "$SMOKE_TEST" \
      --stratified-seed "$SEED" \
      --output results_v8_smoke_a1.json

  run_step smoke_a2 \
    python train/train_a.py \
      --data "$DATA_DIR" \
      --decoder transformer \
      --batch-size 8 \
      --phase1-epochs 1 \
      --phase2-epochs 0 \
      --stratified-train-samples "$SMOKE_TRAIN" \
      --stratified-val-samples "$SMOKE_VAL" \
      --sample-seed "$SEED" \
      --checkpoint-dir checkpoints_v8_smoke \
      "${WANDB_ARGS[@]}"

  run_step smoke_a2_eval \
    python evaluate/evaluate.py \
      --data "$DATA_DIR" \
      --model a2 \
      --checkpoint checkpoints_v8_smoke/model_a2/best.pt \
      --stratified-limit "$SMOKE_TEST" \
      --stratified-seed "$SEED" \
      --output results_v8_smoke_a2.json

  run_step smoke_b2_qwen25 \
    python train/train_b.py \
      --data "$DATA_DIR" \
      --backend qwen25 \
      --load-in-4bit \
      --batch-size "$B_BATCH_SIZE" \
      --grad-accum-steps "$B_GRAD_ACCUM" \
      --learning-rate "$B_LR" \
      --epochs 1 \
      --stratified-train-samples "$SMOKE_TRAIN" \
      --stratified-val-samples "$SMOKE_VAL" \
      --sample-seed "$SEED" \
      --checkpoint-dir checkpoints_b2_qwen25_v8_smoke \
      "${WANDB_ARGS[@]}"

  run_step smoke_b2_qwen25_eval \
    python evaluate/evaluate.py \
      --data "$DATA_DIR" \
      --model b2 \
      --backend qwen25 \
      --load-in-4bit \
    --lora-path checkpoints_b2_qwen25_v8_smoke/model_b2_qwen25/best_lora \
      --stratified-limit "$SMOKE_TEST" \
      --stratified-seed "$SEED" \
      --output results_b2_qwen25_v8_smoke.json

  if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo "[$(date '+%F %T')] SMOKE ONLY DONE" | tee -a "$LOG_DIR/run.log"
    exit 0
  fi
elif [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
  echo "[$(date '+%F %T')] SMOKE SKIPPED; NOTHING TO DO" | tee -a "$LOG_DIR/run.log"
  exit 0
fi

run_step full_a1 \
  python train/train_a.py \
    --data "$DATA_DIR" \
    --decoder lstm \
    --batch-size "$A_BATCH_SIZE" \
    --phase1-epochs "$A_EPOCHS" \
    --phase2-epochs 0 \
    --stratified-train-samples "$A_SAMPLES" \
    --stratified-val-samples "$A_VAL_SAMPLES" \
    --sample-seed "$SEED" \
    --checkpoint-dir checkpoints_v8_a_50k \
    "${WANDB_ARGS[@]}"

run_step full_a2 \
  python train/train_a.py \
    --data "$DATA_DIR" \
    --decoder transformer \
    --batch-size "$A_BATCH_SIZE" \
    --phase1-epochs "$A_EPOCHS" \
    --phase2-epochs 0 \
    --stratified-train-samples "$A_SAMPLES" \
    --stratified-val-samples "$A_VAL_SAMPLES" \
    --sample-seed "$SEED" \
    --checkpoint-dir checkpoints_v8_a_50k \
    "${WANDB_ARGS[@]}"

run_step full_b2_qwen25 \
  python train/train_b.py \
    --data "$DATA_DIR" \
    --backend qwen25 \
    --load-in-4bit \
    --batch-size "$B_BATCH_SIZE" \
    --grad-accum-steps "$B_GRAD_ACCUM" \
    --learning-rate "$B_LR" \
    --epochs "$B_EPOCHS" \
    --stratified-train-samples "$B_SAMPLES" \
    --stratified-val-samples "$B_VAL_SAMPLES" \
    --sample-seed "$SEED" \
    --checkpoint-dir checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5 \
    "${WANDB_ARGS[@]}"

run_step eval_a1_final \
  python evaluate/evaluate.py \
    --data "$DATA_DIR" \
    --model a1 \
    --checkpoint checkpoints_v8_a_50k/model_a1/best.pt \
    "${EVAL_ARGS[@]}" \
    --output "results_v8_a1_50k_${EVAL_SUFFIX}.json"

run_step eval_a2_final \
  python evaluate/evaluate.py \
    --data "$DATA_DIR" \
    --model a2 \
    --checkpoint checkpoints_v8_a_50k/model_a2/best.pt \
    "${EVAL_ARGS[@]}" \
    --output "results_v8_a2_50k_${EVAL_SUFFIX}.json"

run_step eval_b2_qwen25_final \
  python evaluate/evaluate.py \
    --data "$DATA_DIR" \
    --model b2 \
    --backend qwen25 \
    --load-in-4bit \
    --lora-path checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora \
    "${EVAL_ARGS[@]}" \
    --output "results_b2_qwen25_v8_50k_strat_4bit_lr5e5_${EVAL_SUFFIX}.json"

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG_DIR/run.log"
