# VQA Handoff

Last updated: 2026-04-30

This is the teammate handoff entrypoint for the `vqa/` project.

## Read In This Order

1. `PROJECT_STATE.md`
2. `docs/VAST_AI_RUNBOOK.md`
3. `docs/TRAIN_EVAL_RUNBOOK.md`
4. `docs/RL_AND_IMPROVEMENT_PLAN.md`

## What This Project Is

Vietnamese traffic-sign Visual Question Answering for the Deep Learning final
project.

Required four reported models:

- `A1`: custom CLIP + PhoBERT + co-attention + LSTM decoder
- `A2`: custom CLIP + PhoBERT + co-attention + Transformer decoder
- `B1`: pretrained multimodal zero-shot
- `B2`: pretrained multimodal fine-tuned

## Current Truth

- Active dataset: `data/processed/annotations/`
- Active B backend: `qwen25`
- Active B base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Current best practical B2 checkpoint:
  `checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora`
- Current A recommendation: keep `phase2_epochs=0`

## Existing Results

Canonical full-test result:

- `results_all_final_test.json`
- `predictions_all_final_test.jsonl`

Key numbers:

- `A1`: `0.9056` VQA accuracy
- `A2`: `0.9156` VQA accuracy
- `B1`: `0.1967` VQA accuracy
- `B2`: `0.7681` VQA accuracy

## What To Transfer

For a real teammate handoff or upload to a rented GPU machine, transfer the full
`vqa/` directory, especially:

- `data/processed/`
- `checkpoints/`
- `checkpoints_b2_qwen25_10k/`
- `docs/`
- `scripts/`
- `train/`
- `evaluate/`
- `models/`
- `results_*.json`
- `predictions_*.jsonl`

Do not transfer `../nlp/` as part of this handoff unless explicitly needed.

## Packaging

Create a zip of the full `vqa/` project:

```bash
cd /home/pc5070ti/workspace/SDA
python vqa/scripts/package_handoff.py --mode full --archive
```

Output:

- folder: `handoff/vqa_full_handoff/`
- archive: `handoff/vqa_full_handoff.zip`

## First Command On The Recipient Machine

After extracting the archive and entering `vqa/`:

```bash
python -m pip install -r requirements.txt
```

Then follow `docs/VAST_AI_RUNBOOK.md`.

