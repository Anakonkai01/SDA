# Claude Code Context: VQA

Read this before making changes in `vqa/`.

## Active Project

This repository is currently focused on Vietnamese traffic sign VQA. Do not switch back to the NLP legal project unless the user explicitly asks.

Canonical state file:

- `vqa/PROJECT_STATE.md`

Requirement spec:

- `vqa/docs/DỰ ÁN CUỐI KỲ MÔN HỌC SÂU.docx`

Current handoff/data notes:

- `vqa/README_HANDOFF.md`

## Current Data Choice

Use `vqa/data/processed/annotations/` for training and evaluation unless the user explicitly asks for another split.

The active dataset is the V5 group-fix Drive dataset:

- Train: 104146 QA over 2193 images
- Val: 12944 QA over 272 images
- Test: 12966 QA over 271 images

Keep `vqa/data/processed/annotations_rulebased_v5_groupfix/` as the canonical same-content copy.

## Important Constraints

- Split must be by image, not by QA row.
- Test images must not overlap train images.
- Answers should stay short, normally no more than 10 words.
- Final report must compare A1, A2, B1, and B2.
- Required metrics include VQA Accuracy, BLEU, ROUGE-L, METEOR, BERTScore, and LLM-as-a-judge.

## Known Gaps

- METEOR is implemented; optional BERTScore is exposed through `--bertscore`; LLM-as-a-judge still needs implementation.
- A1/A2/B1/B2 smoke paths passed, but they need real training/evaluation results.
- Test split manual review should be documented before final submission.
- For smoke runs, use train/evaluate subset flags instead of launching full training immediately.
