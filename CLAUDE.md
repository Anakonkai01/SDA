# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Also read `AGENTS.md` for newer operational notes, current experiment status, checkpoints, results, and gotchas before making project decisions.

## Repository shape

This repo contains two separate Python research/course projects, not one shared application:

- `vqa/`: Vietnamese traffic-sign Visual Question Answering for the Deep Learning course.
- `nlp/`: Vietnamese traffic-law QA with local-text-only RAG and LoRA fine-tuning for the NLP course.

Work from the relevant subdirectory. Do not touch the other project unless the user asks for cross-project work.

## Environment and dependencies

There is no CI, Dockerfile, or package manager metadata at the repo root. Install dependencies per project:

```bash
cd vqa
pip install -r requirements.txt

cd nlp
pip install -r requirements.txt
```

Both projects are GPU-oriented Python codebases. Check CUDA with:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

## VQA project

Read `vqa/PROJECT_STATE.md` first when working under `vqa/`; `vqa/CLAUDE.md` says it is the canonical current context. Useful handoff/runbook docs are `vqa/README_HANDOFF.md`, `vqa/docs/TRAIN_EVAL_RUNBOOK.md`, and `vqa/docs/VAST_AI_RUNBOOK.md`.

### Architecture

The VQA project compares four model configurations over Vietnamese traffic-sign image/question pairs:

- A1: CLIP ViT-B/16 image encoder + PhoBERT text encoder + co-attention + LSTM decoder.
- A2: same dual encoder/co-attention stack with a Transformer decoder.
- B1: `Qwen/Qwen2.5-VL-3B-Instruct` zero-shot.
- B2: Qwen2.5-VL with LoRA fine-tuning.

Core code is split by function:

- `vqa/models/`: model definitions for A/B routes and attention/decoder components.
- `vqa/data_utils/`: dataset and collator logic.
- `vqa/train/`: training entrypoints and dataclass configs.
- `vqa/evaluate/`: evaluation entrypoints and metrics.
- `vqa/scripts/`: dataset generation, validation, preference/DPO utilities, and audits.
- `vqa/demo/app.py`: Gradio demo.

Operational defaults from project docs:

- Use `data/processed/annotations/` for normal train/eval.
- Use Qwen2.5-VL as the active B backend; BLIP in `model_b.py` is legacy and PaliGemma is only a pilot.
- Use `--backend qwen25 --load-in-4bit` for B models.
- Treat `checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora` as the default B2 checkpoint unless a newer one is explicitly validated.
- Use `--max-pixels 501760` for Qwen B-model evaluation/training when available.
- Keep A1/A2 phase 2 disabled (`--phase2-epochs 0`) unless deliberately experimenting with unfreezing.
- Split and evaluate by image ID, not by arbitrary QA rows; use stratified limits for fair subset checks.

### Common VQA commands

Run these from `vqa/`.

Validate the processed dataset:

```bash
python scripts/validate_dataset.py \
  --annotations_dir data/processed/annotations \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --out data/processed/metadata/validation_report.json
```

Smoke train A1:

```bash
python train/train_a.py \
  --decoder lstm \
  --data data/processed/annotations \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --batch-size 4 \
  --num-workers 0 \
  --phase1-epochs 1 \
  --phase2-epochs 0 \
  --checkpoint-dir checkpoints_smoke \
  --no-mixed-precision
```

Train A1/A2 normally:

```bash
python train/train_a.py --decoder lstm --data data/processed/annotations --phase2-epochs 0
python train/train_a.py --decoder transformer --data data/processed/annotations --phase2-epochs 0
```

Train B2 LoRA:

```bash
python train/train_b.py --backend qwen25 --load-in-4bit --data data/processed/annotations
```

Smoke evaluate one model:

```bash
python evaluate/evaluate.py \
  --model a1 \
  --checkpoint checkpoints_smoke/model_a1/best.pt \
  --data data/processed/annotations \
  --limit 32 \
  --output results_smoke_a1.json
```

Evaluate Qwen B2 on a balanced subset:

```bash
python evaluate/evaluate.py --model b2 --backend qwen25 --load-in-4bit --stratified-limit 1000
```

Launch demo:

```bash
python demo/app.py
```

## NLP project

The NLP project implements a local-text-only traffic-law QA pipeline:

```text
filter traffic laws -> build KB -> generate QA -> fine-tune -> evaluate -> demo
```

The production path must read only `.txt` law sources enabled in `nlp/docs/docs_giaothong/manifest.json` with `source_policy: local_text_only`. Do not add PDF fallback, VLSP corpus fallback, or auto-crawling to production paths unless explicitly requested.

### Architecture

- `nlp/docs/docs_giaothong/text/`: UTF-8 source law text files.
- `nlp/docs/docs_giaothong/manifest.json`: source allowlist for KB builds.
- `nlp/src/corpus.py`: loads local text corpus according to the manifest policy.
- `nlp/src/chunking.py`: splits legal documents for retrieval.
- `nlp/src/build_kb.py`: builds the FAISS/vector KB and metadata guardrail.
- `nlp/src/retrieval.py`: retrieval/reranking path used by RAG configs.
- `nlp/src/generate_qa.py`: QA dataset generation.
- `nlp/src/finetune.py`: LoRA fine-tuning.
- `nlp/src/evaluate.py` and `nlp/src/evaluate_mc.py`: open-ended and multiple-choice evaluation.
- `nlp/src/app.py`: interactive app/demo.

Important artifacts:

- Vector DB: `nlp/vector_db_traffic/`.
- KB metadata guardrail: `nlp/vector_db_traffic/build_meta.json`.
- QA dataset: `nlp/data/qa_pairs_traffic_v2.jsonl` for the current v2 training path.
- Manual eval: `nlp/data/eval_manual.jsonl`.
- Manual MC eval: `nlp/data/eval_mc_manual.jsonl`.
- Fine-tuned adapters: `nlp/models/qwen3.5-9b-lora-traffic/` and `nlp/models/qwen3.5-9b-lora-traffic-v2/`.
- Reports: `nlp/reports/traffic/`.

`load_vectorstore()` is expected to reject a KB that lacks `build_meta.json` or has the wrong source policy; preserve this guardrail.

### Common NLP commands

Run these from `nlp/`.

Full manual pipeline:

```bash
python scripts/filter_traffic_laws.py
python src/build_kb.py --force
python src/generate_qa.py --force
python scripts/filter_qa.py
python src/finetune.py
python src/evaluate.py --configs A B C D
python src/evaluate_mc.py --configs A B C D
python src/app.py
```

Sequential pipeline:

```bash
REBUILD_KB=1 REGENERATE_QA=1 bash auto_pipeline.sh
```

`auto_pipeline.sh` uses `conda run -n "${CONDA_ENV:-ai}"`; set `CONDA_ENV` if the environment name differs.

QA generation smoke checks:

```bash
python src/generate_qa.py --force --profile smoke
python src/generate_qa.py --force --phases penalties procedures --phase-size 20 --no-negatives --timeout 180 --num-predict 160
python src/generate_qa.py --phases penalties --phase-size 50 --max-workers 1 --max-passes 4 --timeout 180 --num-predict 160
```

Evaluate with optional LLM judge:

```bash
python src/evaluate.py --configs A B C D --judge
```

Run app/demo:

```bash
python src/app.py
```

## Development notes

- There are no discovered pytest, lint, or formatter configs in this repo. Prefer targeted smoke commands above over inventing test commands.
- Large data/model artifacts are present or expected locally; avoid broad `git add .` and avoid committing generated checkpoints, logs, vector DBs, or datasets unless the user explicitly asks.
- API-key-backed generation/evaluation uses environment variables such as `GEMINI_API_KEY` and `OPENROUTER_API_KEY`; do not write secrets into files.
- For UI/demo changes, run the relevant Gradio app and verify the flow in a browser if possible.

## Coding guidelines

- State assumptions before non-trivial implementation; ask if requirements are ambiguous.
- Keep changes surgical: touch only files needed for the user’s request and match existing style.
- Do not refactor adjacent code or add speculative abstractions.
- Remove imports/variables/functions made unused by your own change, but do not delete unrelated dead code unless asked.
- Define a verification goal for multi-step work and run the smallest relevant command that proves the change.
