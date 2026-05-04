# Train / Eval Runbook

Last updated: 2026-04-30

Run commands from `vqa/`.

## Preflight

Check CUDA:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Validate active data:

```bash
python scripts/validate_dataset.py \
  --annotations_dir data/processed/annotations \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --out data/processed/metadata/validation_report.json
```

## A1 / A2 Training

Current recommended setting is phase 1 only.

A1:

```bash
python train/train_a.py \
  --decoder lstm \
  --data data/processed/annotations \
  --batch-size 32 \
  --num-workers 4 \
  --phase1-epochs 15 \
  --phase2-epochs 0 \
  --wandb-run-name a1
```

A2:

```bash
python train/train_a.py \
  --decoder transformer \
  --data data/processed/annotations \
  --batch-size 32 \
  --num-workers 4 \
  --phase1-epochs 15 \
  --phase2-epochs 0 \
  --wandb-run-name a2
```

Resume example:

```bash
python train/train_a.py \
  --decoder transformer \
  --data data/processed/annotations \
  --resume
```

## B1 Zero-Shot Eval

Recommended active B backend:

- `Qwen/Qwen2.5-VL-3B-Instruct`

Optional pilot backend:

- `google/paligemma2-3b-mix-448` with `--backend paligemma2`
- PaliGemma models may require accepting the Gemma/PaliGemma terms on Hugging
  Face or Kaggle before first download.

Smoke:

```bash
python evaluate/evaluate.py \
  --model b1 \
  --backend qwen25 \
  --load-in-4bit \
  --limit 200 \
  --max-pixels 501760 \
  --output results_b1_qwen25_200.json \
  --predictions-output predictions_b1_qwen25_200.jsonl
```

Full test:

```bash
python evaluate/evaluate.py \
  --model b1 \
  --backend qwen25 \
  --load-in-4bit \
  --max-pixels 501760 \
  --eval-batch-size 1 \
  --output results_b1_qwen25_test.json \
  --predictions-output predictions_b1_qwen25_test.jsonl
```

PaliGemma 2 zero-shot stratified pilot:

```bash
python evaluate/evaluate.py \
  --model b1 \
  --backend paligemma2 \
  --load-in-4bit \
  --stratified-limit 200 \
  --output results_b1_paligemma2_mix_strat200.json \
  --predictions-output predictions_b1_paligemma2_mix_strat200.jsonl
```

## B2 Qwen LoRA Training

Current stable practical recipe:

```bash
python train/train_b.py \
  --backend qwen25 \
  --load-in-4bit \
  --batch-size 1 \
  --epochs 1 \
  --max-train-samples 10000 \
  --max-val-samples 1000 \
  --max-pixels 501760 \
  --checkpoint-dir checkpoints_b2_qwen25_10k
```

Notes:

- `train_b.py` uses gradient accumulation from config (`4`) internally
- For `qwen25`, the code forces `num_workers=0`
- `wandb` is enabled by default; add `--no-wandb` only for quick smoke runs

Outputs:

```text
checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora/
checkpoints_b2_qwen25_10k/model_b2_qwen25/last_lora/
```

## B2 PaliGemma 2 Pilot Training

Use this only as a gated pilot. The base checkpoint is
`google/paligemma2-3b-pt-448`; the zero-shot/mix checkpoint is separate and is
not the training base.

```bash
python train/train_b.py \
  --backend paligemma2 \
  --load-in-4bit \
  --batch-size 1 \
  --epochs 1 \
  --max-train-samples 10000 \
  --max-val-samples 1000 \
  --checkpoint-dir checkpoints_b2_paligemma2_10k \
  --wandb-run-name b2_paligemma2_10k
```

Output:

```text
checkpoints_b2_paligemma2_10k/model_b2_paligemma2/best_lora/
checkpoints_b2_paligemma2_10k/model_b2_paligemma2/last_lora/
```

## Open-Ended Evaluation

A1:

```bash
python evaluate/evaluate.py \
  --model a1 \
  --checkpoint checkpoints/model_a1/best.pt \
  --output results_a1_test.json \
  --predictions-output predictions_a1_test.jsonl
```

A2:

```bash
python evaluate/evaluate.py \
  --model a2 \
  --checkpoint checkpoints/model_a2/best.pt \
  --output results_a2_test.json \
  --predictions-output predictions_a2_test.jsonl
```

B2:

```bash
python evaluate/evaluate.py \
  --model b2 \
  --backend qwen25 \
  --load-in-4bit \
  --max-pixels 501760 \
  --lora-path checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora \
  --eval-batch-size 1 \
  --output results_b2_qwen25_test.json \
  --predictions-output predictions_b2_qwen25_test.jsonl
```

B2 stratified 1000-row check:

```bash
python evaluate/evaluate.py \
  --model b2 \
  --backend qwen25 \
  --load-in-4bit \
  --max-pixels 501760 \
  --lora-path checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora \
  --stratified-limit 1000 \
  --output results_b2_qwen25_strat1000.json \
  --predictions-output predictions_b2_qwen25_strat1000.jsonl
```

B2 PaliGemma 2 pilot eval:

```bash
python evaluate/evaluate.py \
  --model b2 \
  --backend paligemma2 \
  --load-in-4bit \
  --lora-path checkpoints_b2_paligemma2_10k/model_b2_paligemma2/best_lora \
  --stratified-limit 1000 \
  --output results_b2_paligemma2_strat1000.json \
  --predictions-output predictions_b2_paligemma2_strat1000.jsonl
```

All four:

```bash
python evaluate/evaluate.py \
  --model all \
  --backend qwen25 \
  --load-in-4bit \
  --max-pixels 501760 \
  --lora-path checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora \
  --eval-batch-size 1 \
  --output results_all_final_test.json \
  --predictions-output predictions_all_final_test.jsonl
```

## Multiple-Choice Eval For B

Build MC data:

```bash
python scripts/build_mc_dataset.py \
  --data data/processed/annotations \
  --split test \
  --output data/processed/mc/test_mc.jsonl
```

B2 MC smoke:

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

B2 MC larger run:

```bash
python evaluate/evaluate_mc.py \
  --mc-data data/processed/mc/test_mc.jsonl \
  --model b2 \
  --backend qwen25 \
  --load-in-4bit \
  --lora-path checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora \
  --limit 1000 \
  --max-pixels 501760 \
  --eval-batch-size 2 \
  --output results_b2_qwen25_mc_1000.json \
  --predictions-output predictions_b2_qwen25_mc_1000.jsonl
```

## Preference / DPO Prep

Build preference pairs after predictions exist:

```bash
python scripts/build_preference_pairs.py --help
python train/train_dpo_b.py --help
```

See `docs/RL_AND_IMPROVEMENT_PLAN.md` before running the RL track.

## Canonical Existing Result Files

- `results_all_final_test.json`
- `predictions_all_final_test.jsonl`
- `results_b1_qwen25_200.json`
- `results_b2_qwen25_200.json`
- `results_b2_qwen25_10k_1000.json`
