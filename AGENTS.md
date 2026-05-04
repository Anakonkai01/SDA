# AGENTS.md

Two course projects in one repo. Pure Python research (no CI, no Docker, no package manager).

## Projects

| Dir | Course | Task |
|-----|--------|------|
| `vqa/` | Học Sâu (7đ + 3đ) | Vietnamese traffic sign VQA — 4 configs (A1/A2/B1/B2) + RL enhancement |
| `nlp/` | Nhập môn Xử lý ngôn ngữ tự nhiên | Traffic-law QA with RAG + LoRA fine-tune (4 configs A/B/C/D) |

## VQA (`vqa/`)

**Entrypoints**: `vqa/PROJECT_STATE.md`, `vqa/docs/TRAIN_EVAL_RUNBOOK.md`, `vqa/CLAUDE.md`

All commands run from `vqa/`.

**Model configs**:

| Config | Arch | Note |
|--------|------|------|
| A1 | CLIP ViT-B/16 + PhoBERT + Co-Attn + LSTM | Phase 2 (unfreezing) disabled |
| A2 | CLIP ViT-B/16 + PhoBERT + Co-Attn + Transformer | Phase 2 disabled |
| B1 | Qwen2.5-VL-3B-Instruct zero-shot | — |
| B2 | Qwen2.5-VL-3B-Instruct + LoRA | Active B backend |

**Latest v8 results** (50k stratified train, full test set):

| Model | VQA Acc | BLEU-4 | ROUGE-L | METEOR | BERTScore | Latency |
|-------|:-------:|:------:|:-------:|:------:|:---------:|:-------:|
| A1 | **0.9484** | **0.9602** | **0.9584** | **0.9558** | 0.9631 | 11.0ms |
| A2 | 0.9377 | 0.9476 | 0.9486 | 0.9454 | **0.9713** | 12.5ms |
| B1 | 0.1962 | 0.0350 | 0.2899 | 0.3017 | 0.4753 | 179ms |
| B2 | 0.9379 | 0.9494 | 0.9508 | 0.9478 | 0.9111 | 484ms |
| B2-DPO (100p) | 0.755† | 0.5705 | 0.7939 | 0.7874 | — | 216ms |

† Stratified 1000. Full-test DPO failed due to mode collapse.

**DPO findings**: 100 pairs valid (+1.5%, +20% on negative), 3146 pairs mode-collapsed → `"` token. Theory: $N_{critical} \approx V \cdot \epsilon_{4bit} / (\eta \cdot f_{collapse}) = 2,968$ pairs. Mitigation: lr=1e-6 prevents collapse. 12 checkpoints saved at `checkpoints_b2_dpo_full/checkpoint_*pairs/`.

**Training curves**: wandb logs at `vqa/wandb/run-20260501_034204` (A1 v8), `run-20260501_045024` (A2 v8), `run-20260501_055843` (B2 v8). Charts generated in `vqa/docs/figures/`.

**Report**: `vqa/docs/VQA_REPORT.md` (758 lines, 3 mermaid diagrams, 8 matplotlib charts, 30+ math formulas).

**v5 vs v8 improvements:**

| Model | Acc Δ | BLEU-4 Δ |
|-------|:----:|:--------:|
| A1 | +4.3% | +11.2% |
| A2 | +2.2% | +8.6% |
| B2 | +17.0% | +34.1% |

**Checkpoints (v8):**
- A1: `checkpoints_v8_a_50k/` (best.pt)
- A2: `checkpoints_v8_a_50k/` (best.pt)
- B2: `checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora`

**Operational defaults**:
- Dataset: `data/processed/annotations/` (130K QA rows, 2736 images, split by image)
- B backend: `Qwen/Qwen2.5-VL-3B-Instruct` (BLIP in `model_b.py` is legacy; `paligemma2` is a gated pilot)
- Default B2 checkpoint: `checkpoints_b2_qwen25_10k/model_b2_qwen25/best_lora` (v5) or `checkpoints_b2_qwen25_v8_50k_strat_4bit_lr5e5/model_b2_qwen25/best_lora` (v8)
- A1/A2 phase 2 disabled (`phase2_epochs=0`) — unfreezing caused instability
- RL track: DPO on B2 only (A1/A2 are architecture baselines)
- BERTScore: not yet computed (`null` in results). LLM-as-a-judge: not yet implemented.

**Gotchas**:
- Always `--load-in-4bit` for B models; full precision won't fit on most GPUs
- `--max-pixels 501760` for Qwen B models
- Qwen training forces `num_workers=0` internally
- `train_b.py` uses `grad_accum=4` from config, not cmdline
- Split by image ID, not QA row — test images must not overlap train
- A model image projection is identity (`clip_dim == dim == 768`)
- `--stratified-limit` for fair subset checks; don't compare first-N slices
- PaliGemma: accept HF terms before first download
- API keys: copy `vqa/.env.example` to `.env`, fill `GEMINI_API_KEY`, `OPENROUTER_API_KEY`
- DPO training (`train_dpo_b.py`) is BLIP-specific — needs update for qwen25 backend if used

**Required comparisons** (from đề bài):
- A1 vs A2 (effect of LSTM vs Transformer decoder)
- B1 (zero-shot) vs B2 (fine-tuned) — B1 v8 eval done
- All 4 on VQA Accuracy, BLEU, ROUGE-L, METEOR, BERTScore, LLM-as-a-judge — BERTScore done, Judge tạm dừng
- RL (DPO) vs SFT on B2 with ≥100 preference pairs — **Đã chạy DPO** (100 pairs, accuracy +1.5%)

**DPO result** (100 val preference pairs, stratified 1000 test):
| Metric | SFT | DPO | Δ |
|--------|:---:|:---:|:-:|
| Accuracy | 0.740 | **0.755** | +1.5% |
| negative subtype | 0.336 | **0.536** | +20.0% |
| location | 0.840 | **0.888** | +4.8% |
Checkpoint: `checkpoints_b2_dpo/best_lora`
- RL (DPO) vs SFT on B2 with ≥100 preference pairs — **Đã chạy DPO** (100 pairs, accuracy +1.5%)

**DPO result** (100 val preference pairs, stratified 1000 test):
| Metric | SFT | DPO | Δ |
|--------|:---:|:---:|:-:|
| Accuracy | 0.740 | **0.755** | +1.5% |
| negative subtype | 0.336 | **0.536** | +20.0% |
| location | 0.840 | **0.888** | +4.8% |
Checkpoint: `checkpoints_b2_dpo/best_lora`

**Eval optimization**: `--eval-batch-size 16` (mặc định mới) thay vì 1 giảm latency ~30% (337→231ms/sample). Ảnh test đều 960x540 nên `image_grid_thw` đồng nhất — batch inference an toàn. Image cache (`Image.open` chỉ 1 lần cho mỗi ảnh unique) cũng được bật tự động.

## NLP (`nlp/`)

**Entrypoint**: `nlp/README.md`

Pipeline: filter traffic laws → build KB → generate QA → fine-tune → eval (4 configs).

**Workflow**:
```bash
cd nlp
python scripts/filter_traffic_laws.py
python src/build_kb.py --force
python src/generate_qa.py --force
python scripts/fix_qa_dataset.py                # post-process: reduce negatives + hard-context
python src/finetune.py                          # trains v2 model
python src/evaluate.py --configs A B C D        # uses eval_manual.jsonl (187 clean samples)
python src/evaluate.py --configs A B C D --judge # + LLM-Judge via OpenRouter Gemini
python src/evaluate_mc.py --configs A B C D
python src/app.py
```

Or sequential: `REBUILD_KB=1 REGENERATE_QA=1 bash nlp/auto_pipeline.sh`

**Configs to compare**:

| | No RAG | RAG |
|---|---|---|
| Base LLM | A | B |
| Fine-tuned | C | D |

**NLP v2 retrain** (May 3, 2026 — fix over-refusal):
- Root cause: 44.6% negative ratio → 83.4% false refusal in Config C
- Fix: negatives reduced to 200 (8.6%), +200 hard-context examples (+ "(Kiến thức chung)" marker), +326 additional positives, LoRA rank 16→32, LR 1e-4, epochs 3, context dropout 90/10
- **Config C v2**: ROUGE-L **0.4459** (+38%), BLEU **0.2224** (+85%), **0% false refusal** (was 83.4%), Legacy 0%, LLM-Judge 2.05/5
- **Config D v2**: ROUGE-L 0.3662, BLEU 0.1947 (+22%), **10.4% false refusal** (was 71.3%), Legacy 0%
- V2 model: `models/qwen3.5-9b-lora-traffic-v2/` (best epoch 2, eval_loss 0.1779), v1: `models/qwen3.5-9b-lora-traffic/`
- LLM-Judge: OpenRouter Gemini 2.0 Flash (not critical — ROUGE/BLEU/FRef tell the story)
- Retrieval v3: multi-hop (article-aware re-query) + 30 new query-to-legal mappings. Marginally improved Recall@5.
- Key lesson: more hard-context examples (500 = 18.6%) degraded D's FRef (10.4% → 35.7%). 200 (10%) was the sweet spot.

- Default demo: Config D (fine-tuned + RAG)
- LLM: `Qwen/Qwen3.5-9B` with LoRA
- Local-text-only policy — reads only `.txt` files enabled in `docs/docs_giaothong/manifest.json`
- No corpus VLSP, no PDF fallback, no auto-crawl
- KB metadata guardrail: refuses KB missing `build_meta.json` or wrong `source_policy`
- Eval reports `forbidden_legacy_rate` (citations to old laws like Luật 2008, NĐ 100/2019)

**Gotchas**:
- Source texts go in `docs/docs_giaothong/text/`, enable in `manifest.json`
- Vector DB: `vector_db_traffic/`, KB meta: `vector_db_traffic/build_meta.json`
- QA dataset: `data/qa_pairs_traffic_v2.jsonl` (v2, 2,325 samples, 82.8% positive), manual eval: `data/eval_manual.jsonl` (187 samples, cleaned)
- Fine-tuned adapter v1: `models/qwen3.5-9b-lora-traffic/`, v2: `models/qwen3.5-9b-lora-traffic-v2/`
- Retrieval v2: FAISS semantic → lexical re-rank → cross-encoder (BAAI/bge-reranker-v2-m3). Requires `pyvi` for VN word segmentation.
- Recall@5 improved from 0.27→0.42 after retrieval v2 rewrite.
- Cross-encoder reranker loads on first query; reduce `candidate_k=15` in `retrieval.py` if latency matters.
- 4 new docs needed: `qcvn_41_2024_bgtvt`, `tt_05_2025_tt_bgtvt`, `nd_39_2025_nd_cp`, `tt_79_2024_tt_bca` — disabled pending download.
- QA filtering: `scripts/filter_qa.py` removes answers >500 chars + Điều-template questions. Run before fine-tune. Report saved to `reports/traffic/qa_filter_report.json` for use in final report.

## Quick verification

```bash
# VQA dataset validation (from vqa/)
python scripts/validate_dataset.py \
  --annotations_dir data/processed/annotations \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --out data/processed/metadata/validation_report.json

# Check CUDA
python -c 'import torch; print(torch.cuda.is_available())'
```

---
# CORE CODING PHILOSOPHY (KARPATHY STYLE)
Mọi Agent định nghĩa ở trên PHẢI tuân thủ các nguyên tắc dưới đây trong mọi phản hồi và thao tác file.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
