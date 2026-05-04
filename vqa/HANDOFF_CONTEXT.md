# VQA Handoff Context

Last updated: 2026-04-30

## Canonical Files

Read these first:

1. `PROJECT_STATE.md`
2. `README_HANDOFF.md`
3. `docs/VAST_AI_RUNBOOK.md`
4. `docs/TRAIN_EVAL_RUNBOOK.md`

## Scope

This handoff is only for `vqa/`, the Vietnamese traffic-sign Visual Question
Answering project. Ignore `../nlp/`.

## Active Dataset

- `data/processed/annotations/`
- Train `104146`
- Val `12944`
- Test `12966`
- Images used by annotations: `2736`

## Required Reported Models

- `A1`: custom CLIP + PhoBERT + co-attention + LSTM decoder
- `A2`: custom CLIP + PhoBERT + co-attention + Transformer decoder
- `B1`: Qwen2.5-VL zero-shot
- `B2`: Qwen2.5-VL LoRA fine-tuned

## Current Recommendations

- Treat `Qwen/Qwen2.5-VL-3B-Instruct` as the active B model.
- Treat `checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora` as the default
  B2 checkpoint.
- Keep `A1/A2` phase 2 disabled unless intentionally re-experimenting.
- Use open-ended eval as the main official comparison.
- Use MC eval only as auxiliary analysis for B models.

## Current Full-Test Metrics

From `results_all_final_test.json`:

- `A1_LSTM`: accuracy `0.9056`
- `A2_Transformer`: accuracy `0.9156`
- `B1_Qwen25_ZeroShot`: accuracy `0.1967`
- `B2_Qwen25_LoRA`: accuracy `0.7681`

## Existing Artifacts Worth Preserving

- `results_all_final_test.json`
- `predictions_all_final_test.jsonl`
- `results_b2_qwen25_10k_1000.json`
- `predictions_b2_qwen25_10k_1000.jsonl`
- `b2_wrong_review_200.csv`
- `b2_wrong_review_200.html`

## If Continuing Work

Next practical direction:

1. Reconfirm runs on the new GPU machine
2. Run B1/B2 MC evaluation at larger scale
3. Build reviewed preference pairs
4. Run DPO on B2
5. Add LLM-as-a-judge or human preference comparison

