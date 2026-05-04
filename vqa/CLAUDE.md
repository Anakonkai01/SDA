# VQA Working Context

Read `PROJECT_STATE.md` first. It is the canonical current context.

This directory is the active Vietnamese traffic-sign VQA project. Do not touch
`../nlp/` unless explicitly asked.

Operational defaults:

- Use `data/processed/annotations/` for normal train/eval
- Treat `Qwen/Qwen2.5-VL-3B-Instruct` as the active B backend
- Treat `checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora` as the default
  B2 checkpoint unless a newer one is explicitly validated
- Treat `A1/A2` phase 2 as disabled unless there is a deliberate experiment to
  revisit unfreezing

If handing the project to another person or machine, start from:

- `README_HANDOFF.md`
- `docs/VAST_AI_RUNBOOK.md`
- `docs/TRAIN_EVAL_RUNBOOK.md`

