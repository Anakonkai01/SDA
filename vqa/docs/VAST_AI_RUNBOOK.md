# Vast.ai Runbook

Last updated: 2026-04-30

This runbook is for handing `vqa/` to another person who will run it on a
remote GPU, especially an RTX `5090` on Vast.ai.

## 1. Transfer

On the source machine:

```bash
cd /home/pc5070ti/workspace/SDA
python vqa/scripts/package_handoff.py --mode full --archive
```

Send:

- `handoff/vqa_full_handoff.zip`

## 2. Prepare The Vast.ai Instance

Recommended:

- CUDA-capable image with Python `3.11` or `3.12`
- Enough disk for:
  - full `vqa/` project
  - Hugging Face model cache
  - checkpoints
  - outputs

After upload/extract:

```bash
cd /workspace/vqa
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Sanity Check

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

## 4. Validate Data

```bash
python scripts/validate_dataset.py \
  --annotations_dir data/processed/annotations \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --out data/processed/metadata/validation_report.json
```

## 5. First Useful Commands

Open-ended B2 smoke:

```bash
python evaluate/evaluate.py \
  --model b2 \
  --backend qwen25 \
  --load-in-4bit \
  --limit 32 \
  --max-pixels 501760 \
  --lora-path checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora \
  --output /tmp/results_b2_smoke.json \
  --predictions-output /tmp/preds_b2_smoke.jsonl
```

MC B2 smoke:

```bash
python evaluate/evaluate_mc.py \
  --mc-data data/processed/mc/test_mc.jsonl \
  --model b2 \
  --backend qwen25 \
  --load-in-4bit \
  --lora-path checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora \
  --limit 32 \
  --max-pixels 501760 \
  --eval-batch-size 1 \
  --output /tmp/results_b2_mc_smoke.json \
  --predictions-output /tmp/preds_b2_mc_smoke.jsonl
```

## 6. Recommended Real Runs

### A2

```bash
python train/train_a.py \
  --decoder transformer \
  --data data/processed/annotations \
  --batch-size 32 \
  --num-workers 4 \
  --phase1-epochs 15 \
  --phase2-epochs 0 \
  --wandb-run-name a2_vast
```

### B2

```bash
python train/train_b.py \
  --backend qwen25 \
  --load-in-4bit \
  --batch-size 1 \
  --epochs 1 \
  --max-train-samples 10000 \
  --max-val-samples 1000 \
  --max-pixels 501760 \
  --checkpoint-dir checkpoints_b2_qwen25_10k \
  --wandb-run-name b2_qwen25_10k_vast
```

## 7. Existing Result Anchors

Use these files to confirm the transferred project state:

- `PROJECT_STATE.md`
- `results_all_final_test.json`
- `predictions_all_final_test.jsonl`

## 8. Important Practical Notes

- The active B path is Qwen, not BLIP.
- A phase 2 is intentionally disabled in the current baseline.
- Full open-ended B eval is slow even when VRAM usage looks low.
- MC eval is auxiliary analysis, not a replacement for the required open-ended
  final comparison.
- If using `wandb`, run `wandb login` first.

## 9. If The Recipient Wants To Continue RL Work

Start from:

- `docs/RL_AND_IMPROVEMENT_PLAN.md`
- `scripts/build_preference_pairs.py`
- `train/train_dpo_b.py`

