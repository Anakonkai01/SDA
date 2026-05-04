# VQA Project State

Last updated: 2026-04-30

## Scope

This directory is the active Vietnamese traffic-sign VQA project.

Do not touch `../nlp/` unless explicitly requested. The current handoff is for
`vqa/` only.

## Canonical Context

- Project requirement anchor: `docs/DỰ ÁN CUỐI KỲ MÔN HỌC SÂU.docx`
- Teammate handoff entry: `README_HANDOFF.md`
- Vast.ai setup/runbook: `docs/VAST_AI_RUNBOOK.md`
- Train/eval commands: `docs/TRAIN_EVAL_RUNBOOK.md`
- Advanced RL direction: `docs/RL_AND_IMPROVEMENT_PLAN.md`
- Working-agent pointer: `CLAUDE.md`

## Goal

Build a Vietnamese Visual Question Answering system for traffic signs.

Input:

- One street image
- One Vietnamese question

Output:

- One short Vietnamese answer

Required comparison:

- `A1`: custom model with LSTM decoder
- `A2`: custom model with Transformer decoder
- `B1`: pretrained multimodal zero-shot
- `B2`: pretrained multimodal fine-tuned

## Active Dataset

Use:

```text
data/processed/annotations/
```

Current validated counts:

- Train: `104146`
- Val: `12944`
- Test: `12966`
- Total: `130056`

Image counts:

- Train: `2193`
- Val: `272`
- Test: `271`
- Total: `2736`

Other active dataset assets:

- Images: `data/processed/images/`
- Metadata: `data/processed/metadata/`
- Canonical same-content copy: `data/processed/annotations_rulebased_v5_groupfix/`
- Review CSV: `data/processed/review/review_rulebased_archive_all_stratified_v5_groupfix.csv`

The current dataset came from Google Drive file ID
`1NmOYhTeuF5q-GVt4EIMlg9Axd9hRDG3X` and passed repo validation with
`errors=0`, `warnings=0`.

## Implemented Model Paths

### A route

- Image encoder: `openai/clip-vit-base-patch16`
- Text encoder: `vinai/phobert-base`
- Fusion: co-attention
- A1 decoder: LSTM
- A2 decoder: Transformer

Important implementation note:

- CLIP patch features are already `768`, so image projection is identity when
  `clip_dim == dim`.

### B route

The active B route is now `Qwen/Qwen2.5-VL-3B-Instruct`, not BLIP.

- `B1`: Qwen2.5-VL zero-shot
- `B2`: Qwen2.5-VL + LoRA
- Quantization path used in practice: `--load-in-4bit`
- Default B2 checkpoint priority:
  - `checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora`
  - fallback `checkpoints/model_b2_qwen25/best_lora`

`models/model_b.py` still supports `blip`, but current experiments, results,
and handoff assume `backend=qwen25`.

Optional pilot backend:

- `paligemma2` is implemented for gated experiments, not as the canonical B
  route.
- Zero-shot/default exploration model: `google/paligemma2-3b-mix-448`
- LoRA fine-tune base model: `google/paligemma2-3b-pt-448`
- Pilot checkpoints should use `checkpoints_b2_paligemma2_10k/`.
- Use `evaluate/evaluate.py --stratified-limit` for fair small-subset checks;
  avoid comparing against first-N test slices.

## Training State

### A1/A2

Current config in `train/config.py`:

- `batch_size = 32`
- `epochs = 15`
- `phase1_epochs = 15`
- `phase2_epochs = 0`

Reason:

- Unfreezing phase 2 caused instability and poor results for A2.
- The current recommendation is to keep phase 2 disabled for final baseline
  runs unless there is time for a controlled revisit.

### B2

Current practical B2 route:

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- LoRA
- 4-bit loading
- `max_pixels = 501760`
- `max_length = 1024`

Most important finished B2 checkpoint:

- `checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora`

This is the current reference fine-tuned B2 model.

## Finished Evaluation Results

Canonical full-test result file:

- `results_all_final_test.json`
- Predictions: `predictions_all_final_test.jsonl`

Full test metrics:

- `A1_LSTM`: accuracy `0.9056`, BLEU-4 `0.8478`, ROUGE-L `0.9289`, METEOR `0.9226`
- `A2_Transformer`: accuracy `0.9156`, BLEU-4 `0.8620`, ROUGE-L `0.9366`, METEOR `0.9309`
- `B1_Qwen25_ZeroShot`: accuracy `0.1967`, BLEU-4 `0.0352`, ROUGE-L `0.2906`, METEOR `0.3017`
- `B2_Qwen25_LoRA`: accuracy `0.7681`, BLEU-4 `0.6086`, ROUGE-L `0.8146`, METEOR `0.8027`

Inference speed on full test:

- A1: about `11.38 ms/sample`
- A2: about `12.43 ms/sample`
- B1: about `465.66 ms/sample`
- B2: about `590.17 ms/sample`

Additional B2 subset result:

- `results_b2_qwen25_10k_1000.json`: accuracy `0.884` on `1000` samples

Interpretation:

- A route currently wins by a clear margin on this dataset.
- B2 is usable but still underperforms A1/A2 on exact open-ended evaluation.
- B1 zero-shot is weak but valid as the required pretrained zero-shot baseline.

## Multiple-Choice Auxiliary Eval

An auxiliary MC pipeline was added for analysis of B models:

- Dataset builder: `scripts/build_mc_dataset.py`
- Eval entry: `evaluate/evaluate_mc.py`
- MC data:
  - `data/processed/mc/test_mc.jsonl`
  - `data/processed/mc/val_mc.jsonl`

Latest smoke result:

- `B2_qwen25_MC`: accuracy `0.9375` on `32` samples
- avg inference `267.54 ms/sample`
- invalid outputs: `0`

This MC setting is for analysis and controlled comparison of answer selection.
The required primary report comparison should still use the open-ended A1/A2/B1/B2
results.

## Error Analysis State

Available artifacts:

- `b2_wrong_review_200.csv`
- `b2_wrong_review_200.html`

Observed conclusion from manual inspection:

- Most B2 errors are real grounding/content mistakes, not just metric mismatch.
- Some location/attribute cases are semantically close, but they do not explain
  the full open-ended gap.

## RL / Preference Optimization Direction

Current recommendation:

- Apply preference optimization to `B2` only

Reason:

- The course asks for RL/DPO/RLHF-style improvement.
- B2 is the most practical target because it already uses a pretrained VLM with
  LoRA and supports generation.
- A1/A2 should remain the architecture-comparison baselines.

Current implemented/prepared pieces:

- DPO training entry: `train/train_dpo_b.py`
- Preference builder: `scripts/build_preference_pairs.py`
- Preference inspection tools:
  - `scripts/export_preference_review_csv.py`
  - `scripts/import_preference_review_csv.py`
  - `scripts/inspect_preference_pairs.py`

Preferred data sources for preference pairs:

- Gold answer vs wrong model prediction
- Human-reviewed chosen/rejected pairs
- Hard failures from train/val/test predictions

## What To Run Next

If continuing experimentation:

1. Re-run or confirm full B1/B2 eval on the target machine.
2. Run B1/B2 MC eval on `1000` then full test for analysis.
3. Build at least `100` reviewed preference pairs.
4. Run `B2-SFT` vs `B2-DPO`.
5. Add LLM-as-a-judge or human preference comparison for the final report.

## Files That Matter Most In A Handoff

- `README_HANDOFF.md`
- `PROJECT_STATE.md`
- `docs/VAST_AI_RUNBOOK.md`
- `docs/TRAIN_EVAL_RUNBOOK.md`
- `results_all_final_test.json`
- `predictions_all_final_test.jsonl`
- `checkpoints/`
- `checkpoints_b2_qwen25_10k/`
- `data/processed/`
