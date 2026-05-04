# VQA - Vietnamese Traffic Sign Visual Question Answering

A Visual Question Answering system for Vietnamese traffic signs. Given a street image containing traffic signs and a Vietnamese question, the system generates a Vietnamese answer. This project is part of the **Smart Driving Assistant (SDA)** system and serves as the final project for the Deep Learning course.

## Overview

The system implements and compares **4 model configurations**:

| Config | Architecture | Description |
|--------|-------------|-------------|
| **A1** | CLIP ViT-B/16 + PhoBERT + Co-Attention + LSTM Decoder | Custom dual-encoder with LSTM generation |
| **A2** | CLIP ViT-B/16 + PhoBERT + Co-Attention + Transformer Decoder | Custom dual-encoder with Transformer generation |
| **B1** | Qwen2.5-VL-3B-Instruct (zero-shot) | Pretrained multimodal VLM without fine-tuning |
| **B2** | Qwen2.5-VL-3B-Instruct + LoRA | Same pretrained model fine-tuned with LoRA |

Optional pilot backend:

- `paligemma2`: PaliGemma 2 backend for gated B-route experiments. Use
  `google/paligemma2-3b-mix-448` for zero-shot exploration and
  `google/paligemma2-3b-pt-448` for LoRA fine-tuning.

## Architecture

### Approach A - Dual Encoder + Co-Attention

```
Image (224x224)              Question (Vietnamese)
      |                              |
CLIP ViT-B/16 (frozen)       PhoBERT-base (frozen)
      |                              |
  [B, 197, 768]                 [B, N, 768]
      |                              |
Identity projection                  |
      |                              |
  [B, 197, 768]                 [B, N, 768]
      \                            /
       Co-Attention (2 layers)
      /                            \
image_enriched              text_enriched
      \                            /
        Concat -> memory [B, N+197, 768]
                    |
          A1: LSTM Decoder / A2: Transformer Decoder
                    |
            [B, seq, vocab_size]
```

### Approach B - Qwen2.5-VL

- **B1**: Direct zero-shot inference with `Qwen/Qwen2.5-VL-3B-Instruct`
- **B2**: LoRA fine-tuning (r=16, alpha=32) on traffic sign QA data
- Practical route used in this repo: `--backend qwen25 --load-in-4bit`
- PaliGemma 2 can be tested with `--backend paligemma2`; keep it as a pilot
  unless it beats Qwen B2 on a fixed stratified subset.

## Dataset

### Detection Dataset
- **Source**: Kaggle VNTS / Vietnamese Traffic Signs
- **Link**: https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs
- **License**: CC BY-SA 4.0
- **Original task**: traffic sign detection / recognition with bounding boxes and classes
- **Project usage**: convert object annotations into Vietnamese VQA labels

### VQA Dataset
- **Active version**: V5 group-fix rule-based dataset
- **Processed images used by annotations**: 2,736 images
- **Split**: 2,193 train / 272 val / 271 test images, split by `image_id`
- **QA rows**: 104,146 train / 12,944 val / 12,966 test
- **QA/image**: min 23, mean 47.54, max 96
- **Question types**: `yes_no`, `count`, `sign_type`, `color`, `shape`, `location`, `attribute`, `negative`
- **Test policy**: test QA can be generated from Kaggle images, but must be manually reviewed
- **Format**:
```json
{
  "question_id": "vts_000001_q01",
  "image_id": "vts_000001",
  "image_path": "images/train/vts_000001.jpg",
  "question": "Trong ảnh có biển giới hạn tốc độ không?",
  "answer": "Có",
  "question_type": "yes_no",
  "answer_type": "yes_no",
  "evidence_object_ids": ["vts_000001_o1"],
  "split": "train"
}
```

## Project Structure

```
vqa/
├── models/
│   ├── co_attention.py          # Co-Attention module (2-layer stack)
│   ├── decoder_lstm.py          # LSTM decoder with cross-attention
│   ├── decoder_transformer.py   # Transformer decoder with positional encoding
│   ├── model_a.py               # VQAModelA (A1 + A2 configurations)
│   └── model_b.py               # VQAModelB (B1 zero-shot + B2 LoRA)
├── data_utils/
│   ├── dataset.py               # VQADataset class + data loading
│   └── collator.py              # DataCollator for DataLoader
├── train/
│   ├── config.py                # All hyperparameters (dataclasses)
│   ├── train_a.py               # Training loop for A1/A2 (2-phase)
│   └── train_b.py               # Training loop for B2 (LoRA)
├── evaluate/
│   ├── evaluate.py              # Run evaluation on test set
│   └── metrics.py               # BLEU, ROUGE-L, BERTScore, VQA Accuracy
├── demo/
│   └── app.py                   # Gradio demo interface
├── scripts/
│   ├── prepare_dataset.py       # Prepare Kaggle VNTS images and objects.jsonl
│   ├── generate_vqa_labels.py   # Generate raw Q&A with Gemini or dry-run mock
│   ├── filter_vqa.py            # Normalize/filter raw Q&A and export review.csv
│   ├── build_final_jsonl.py     # Build train/val/test JSONL after review
│   └── validate_dataset.py      # Validate final VQA dataset constraints
├── src/data/                    # Shared data utilities
├── prompts/
│   └── vqa_traffic_vi.txt       # Gemini prompt for VQA labeling
├── data/
│   ├── raw/kaggle_vnts/         # Downloaded Kaggle data, not committed
│   └── processed/               # Images, metadata, review files, annotations
├── docs/                        # Design documents, runbooks, and requirements
│   ├── TRAIN_EVAL_RUNBOOK.md    # Full training/evaluation commands
│   └── RL_AND_IMPROVEMENT_PLAN.md # DPO/PPO/RLHF advanced track
└── requirements.txt
```

## Generate VQA Labels

### 0. Download and prepare Kaggle VNTS

Download the dataset into `data/raw/kaggle_vnts`:

```bash
mkdir -p data/raw/kaggle_vnts
env KAGGLE_CONFIG_DIR=$PWD/data/kaggle_config \
  kaggle datasets download -d maitam/vietnamese-traffic-signs \
  -p data/raw/kaggle_vnts --unzip
```

Prepare 350 images and object metadata:

```bash
python scripts/prepare_dataset.py \
  --raw_dir data/raw/kaggle_vnts/archive \
  --out_dir data/processed \
  --num_images 350 \
  --train_images 280 \
  --val_images 35 \
  --test_images 35 \
  --seed 42 \
  --min_bbox_size 30
```

The label generation pipeline starts from:

```bash
data/processed/metadata/objects.jsonl
```

Each line contains one image, its split, relative image path, and object annotations. The image path is relative to `data/processed`.

### 1. Test with dry-run mock labels

```bash
python scripts/generate_vqa_labels.py \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --prompt prompts/vqa_traffic_vi.txt \
  --out data/processed/review/raw_qa_mock.jsonl \
  --splits train \
  --dry_run \
  --limit 5
```

### 2. Generate sample labels with Gemini

Do not write API keys into files. Set the key only in the shell:

```bash
export GEMINI_API_KEY="your_new_key_here"

python scripts/generate_vqa_labels.py \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --prompt prompts/vqa_traffic_vi.txt \
  --out data/processed/review/raw_qa_sample.jsonl \
  --failed_out data/processed/review/failed_generation_sample.jsonl \
  --splits train \
  --provider gemini \
  --model gemini-2.5-flash \
  --limit 5 \
  --temperature 0.4 \
  --resume
```

### 3. Generate full train/val labels

```bash
python scripts/generate_vqa_labels.py \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --prompt prompts/vqa_traffic_vi.txt \
  --out data/processed/review/raw_qa.jsonl \
  --failed_out data/processed/review/failed_generation.jsonl \
  --splits train val \
  --provider gemini \
  --model gemini-2.5-flash \
  --temperature 0.4 \
  --max_output_tokens 4096 \
  --resume
```

OpenRouter can be used instead of direct Gemini when you want to switch VLMs
behind one OpenAI-compatible API. Start with a 5-image pilot before generating
the full split:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"

python scripts/generate_vqa_labels.py \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --prompt prompts/vqa_traffic_vi.txt \
  --out data/processed/review/raw_qa_openrouter_sample.jsonl \
  --failed_out data/processed/review/failed_generation_openrouter_sample.jsonl \
  --splits train \
  --provider openrouter \
  --model google/gemma-4-26b-a4b-it \
  --limit 5 \
  --temperature 0.2 \
  --max_output_tokens 4096 \
  --resume
```

If the Gemini free-tier quota is too small for the full dataset, use the rule-based
fallback generated from Kaggle bbox/class metadata:

```bash
python scripts/generate_vqa_labels.py \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --prompt prompts/vqa_traffic_vi.txt \
  --out data/processed/review/raw_qa_rulebased_v3.jsonl \
  --splits train val test \
  --dry_run
```

### 4. Filter and export review CSV

```bash
python scripts/filter_vqa.py \
  --raw data/processed/review/raw_qa_rulebased_v3.jsonl \
  --out_jsonl data/processed/review/filtered_qa_rulebased_v3.jsonl \
  --out_csv data/processed/review/review_rulebased_v3.csv \
  --max_per_image 10 \
  --min_per_image 3 \
  --max_yes_no_ratio 0.35
```

Open `data/processed/review/review_rulebased_v3.csv` in Excel or Google Sheets.
Set `keep=0` for wrong rows, fill `corrected_question` or `corrected_answer`
when needed, then save as `data/processed/review/reviewed.csv`.

### 5. Build final JSONL

```bash
python scripts/build_final_jsonl.py \
  --review_csv data/processed/review/review_rulebased_v3.csv \
  --out_dir data/processed/annotations \
  --source kaggle_vnts \
  --label_source rule_based_auto_filtered
```

After manual review, replace `--review_csv` with `data/processed/review/reviewed.csv`
and `--label_source` with `rule_based_human_reviewed`.

```bash
python scripts/build_final_jsonl.py \
  --review_csv data/processed/review/reviewed.csv \
  --out_dir data/processed/annotations \
  --source kaggle_vnts \
  --label_source rule_based_human_reviewed
```

### 6. Validate final dataset

```bash
python scripts/validate_dataset.py \
  --annotations_dir data/processed/annotations \
  --objects data/processed/metadata/objects.jsonl \
  --processed_dir data/processed \
  --out data/processed/metadata/validation_report.json
```

## Training

### A1/A2 - Two-Phase Training
```bash
# Train A1 (LSTM decoder)
python train/train_a.py --decoder lstm --data data/processed/annotations

# Train A2 (Transformer decoder)
python train/train_a.py --decoder transformer --data data/processed/annotations
```

For a quick smoke run on the large V5 dataset:

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

Training uses a 2-phase strategy:
- **Phase 1** (epochs 1-15): Frozen CLIP + PhoBERT encoders, lr=3e-4
- **Phase 2** (epochs 16-30): Unfrozen encoders with discriminative lr (encoders: 3e-5, decoder: 3e-4)

### B2 - LoRA Fine-tuning
```bash
python train/train_b.py --data data/processed/annotations --no-bf16
```

LoRA config: r=16, alpha=32, targets=`[query, key, value]`, effective batch=32

PaliGemma 2 pilot:

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

## Evaluation

```bash
# Evaluate single model
python evaluate/evaluate.py --model a1 --checkpoint checkpoints/model_a1/best.pt --data data/processed/annotations

# Evaluate all models
python evaluate/evaluate.py --model all --data data/processed/annotations

# Evaluate B1 zero-shot quickly
python evaluate/evaluate.py --model b1 --data data/processed/annotations --limit 32 --output results_b1_smoke.json --predictions-output predictions_b1_smoke.jsonl

# Evaluate a balanced subset instead of the first N rows
python evaluate/evaluate.py --model b2 --backend qwen25 --load-in-4bit --stratified-limit 1000

# Smoke evaluate on the first 32 test rows
python evaluate/evaluate.py --model a1 --checkpoint checkpoints_smoke/model_a1/best.pt --data data/processed/annotations --limit 32 --output results_smoke_a1.json
```

**Metrics**: BLEU-1, BLEU-4, ROUGE-L, METEOR, exact-match VQA Accuracy, inference time.
`BERTScore` is optional because it is expensive on the full test split:

```bash
python evaluate/evaluate.py --model b1 --data data/processed/annotations --limit 128 --bertscore
```

## Demo

```bash
# Task 1 — VQA demo (port 7860)
python demo/app.py

# Task 2 — Stable Diffusion text-to-image demo (port 7861)
python stable_diffusion/app.py
```

Task 1 launches a Gradio interface where users upload a street image, type a Vietnamese question, and select any of the 4 model configurations (A1/A2/B1/B2) to get an answer.

Task 2 launches a text-to-image Gradio interface backed by Stable Diffusion, with an optional LoRA adapter fine-tuned on Vietnamese traffic sign images.

---

## Task 2: Text-to-Image Generation (Stable Diffusion)

An independent Deep Learning task: given a text prompt, generate a street/traffic-sign image using Stable Diffusion.

- **Model:** `runwayml/stable-diffusion-v1-5` (pretrained) + optional LoRA fine-tuned on 1,500 Vietnamese traffic sign images (12,000 steps, RTX 5070 Ti)
- **Demo:** Gradio app at `http://127.0.0.1:7861` — controls for prompt, negative prompt, steps, guidance scale, seed, size, base vs LoRA selector
- **LoRA adapter:** `stable_diffusion/lora_output/pytorch_lora_weights.safetensors` (6.2 MB)
- **Report:** [`docs/BAI_2_STABLE_DIFFUSION_REPORT.md`](docs/BAI_2_STABLE_DIFFUSION_REPORT.md)
- **Module README:** [`stable_diffusion/README.md`](stable_diffusion/README.md)

```bash
cd vqa
pip install -r requirements.txt
python stable_diffusion/app.py
# → http://127.0.0.1:7861
```

Sample prompts:
```
a realistic Vietnamese street intersection with traffic signs, daytime, high detail
a road safety education poster about Vietnamese traffic signs, clean composition
```

---

## Reports

| Task | Report |
|---|---|
| Task 1 — VQA (Vietnamese) | [`docs/VQA_REPORT_VI_DEEP.md`](docs/VQA_REPORT_VI_DEEP.md) |
| Task 1 — VQA (English) | [`docs/VQA_REPORT_EN_DEEP.md`](docs/VQA_REPORT_EN_DEEP.md) |
| Task 2 — Stable Diffusion | [`docs/BAI_2_STABLE_DIFFUSION_REPORT.md`](docs/BAI_2_STABLE_DIFFUSION_REPORT.md) |
| Slide deck (both tasks) | [`docs/SLIDE_DECK_FULL.md`](docs/SLIDE_DECK_FULL.md) |

---

## Results Summary

| Model | VQA Accuracy | BLEU-4 | ROUGE-L | BERTScore | Latency |
|---|---:|---:|---:|---:|---:|
| A1 LSTM | **0.9484** | **0.9602** | **0.9584** | 0.9631 | ~11 ms |
| A2 Transformer | 0.9377 | 0.9476 | 0.9486 | **0.9713** | ~13 ms |
| B1 Zero-shot | 0.1962 | 0.0350 | 0.2899 | 0.4753 | ~167 ms |
| B2 QLoRA SFT | 0.9379 | 0.9494 | 0.9508 | 0.9111 | ~484 ms |

Human evaluation (100 natural questions, 21 test images): B2-DPO **73%** > B2-SFT 70% > A1 61% > A2 57% > B1 38%.

DPO experiment: 500+ preference pairs, compared RL alignment vs SFT on stratified-1000 subset (0.740 → 0.759 overall, with per-type trade-offs analyzed).

## Requirements

- Python 3.x
- NVIDIA GPU with >= 16GB VRAM (tested on RTX 5070 Ti)
- CUDA 13.1+

```bash
pip install -r requirements.txt
```

## Key Design Decisions

- **CLIP ViT-B/16 over ResNet50**: Preserves spatial information via 197 patch tokens (14x14 grid + CLS), enabling spatial and counting questions
- **Co-Attention over concatenation**: Enables bidirectional cross-modal interaction (text queries image, image queries text)
- **Text generation over classification**: Answers are free-form Vietnamese text, not a fixed vocabulary
- **BLIP VQA for B1/B2 first pass**: Most reliable pretrained multimodal route for getting all four required configs running before deeper model comparison
- **LoRA r=16**: Optimal balance between capacity and VRAM usage for domain-specific adaptation
