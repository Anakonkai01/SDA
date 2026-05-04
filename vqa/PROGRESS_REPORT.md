# Progress Report: Vietnamese Traffic Sign VQA

Last updated: 2026-04-29 (session 2)

## Current Direction

The active project is VQA for Vietnamese traffic signs. The old NLP legal project is not part of the active scope.

Use `PROJECT_STATE.md` as the canonical short context for future chatbot sessions.

## Completed

- Read the course requirement document in `docs/DỰ ÁN CUỐI KỲ MÔN HỌC SÂU.docx`.
- Merged the VQA handoff data/code into `vqa/`.
- Established the current dataset source as Kaggle VNTS converted into VQA.
- Built/kept the rule-based dataset versions under `data/processed/`.
- Verified dataset validation for the current annotation folders.
- Verified Python compilation for the VQA scripts and data helpers.
- Confirmed the repo has code paths for A1, A2, B1, B2.

## Current Dataset

- Current training/evaluation dataset: `data/processed/annotations/` with 104146 train, 12944 val, 12966 test QA rows.
- Canonical same-content copy: `data/processed/annotations_rulebased_v5_groupfix/`.
- Active dataset uses 2193 train, 272 val, 271 test annotated images.
- Previous V3 dataset was moved to `../handoff/processed_backup_before_v5_20260429/`.

## Candidate New Dataset

- Downloaded from Drive file ID `1NmOYhTeuF5q-GVt4EIMlg9Axd9hRDG3X`.
- Local archive: `../handoff/new_vqa_data_drive_file`.
- Extracted root: `../handoff/new_vqa_data/processed_archive_all_stratified_801010/`.
- Recommended folder: `annotations_rulebased_archive_all_stratified_v5_groupfix/`.
- V5 validation passed with 104146 train, 12944 val, 12966 test QA rows.
- V5 uses 2193 train, 272 val, 271 test annotated images with no image overlap across splits.
- Adopted as the active dataset. Only referenced images were copied because the archive has extra unreferenced image files.

## Implemented

- A1/A2 custom architecture: CLIP ViT image encoder, PhoBERT text encoder, co-attention, LSTM or Transformer decoder.
- CLIP ViT-B/16 patch tokens are 768-d, matching PhoBERT/fusion dim, so the image projection is `Identity` unless a future encoder has a different hidden size.
- B1/B2 multimodal path: BLIP VQA zero-shot and LoRA fine-tuning wrapper.
- Training scripts for A1/A2 and B2.
- **wandb integration** in `train/train_a.py` and `train/train_b.py`. Project: `sda-vqa`. Flags: `--wandb-project`, `--wandb-run-name`, `--no-wandb`. Metrics: `train/loss`, `val/loss`, `train/lr`, `phase` (A only), `best_val_loss`, `best_epoch`.
- Evaluation entrypoint for A1/A2/B1/B2.
- Evaluation can save prediction-level JSONL with `--predictions-output` for later error analysis, LLM judging, and RL preference construction.
- Added `docs/TRAIN_EVAL_RUNBOOK.md` for real training/evaluation commands.
- Added `docs/RL_AND_IMPROVEMENT_PLAN.md` for the required PPO/DPO/RLHF advanced track.
- Added `scripts/build_preference_pairs.py` to create preference JSONL from prediction outputs.
- Gradio demo entrypoint.
- Rule-based VQA data generation/filter/build/validate scripts.

## Remaining

- METEOR is now available in the local evaluator.
- Optional BERTScore is available through `evaluate/evaluate.py --bertscore`.
- A1, A2, B1, and B2 smoke paths have been run successfully.
- [ ] Run full training: A1, A2 (30 epochs each), B2 (10 epochs). Commands in `PROJECT_STATE.md`.
- [ ] Run B1 zero-shot evaluation on test set.
- [ ] Build at least 100 preference pairs and compare B2-SFT vs B2-DPO/RL.
- [ ] Add LLM-as-a-judge evaluation.
- [ ] Fill report tables with real metrics from eval outputs.
- [ ] Confirm final test split manual review before report submission.
- [ ] (Optional) Add Qwen2-VL-2B as B3 config for stronger multilingual comparison.

## Design Notes

- BLIP VQA base is English-trained → B1 zero-shot on Vietnamese will score low. This is expected and should be explained in the report as a motivation for B2 LoRA fine-tuning.
- Qwen2-VL was considered as a multilingual alternative for B1/B2 but not implemented. If added, use Qwen2-VL-2B-Instruct (fits 16GB VRAM). Only pursue after A1/A2/B1/B2 results are in hand.

## Cleanup Rule

Keep VQA source, processed annotations, metadata, and final images. Remove or ignore raw downloads, extracted handoff directories, zip packages, smoke-test outputs, and deprecated generated datasets when they are no longer needed.
