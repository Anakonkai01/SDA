# Progress Report: VQA - Vietnamese Traffic Sign Visual Question Answering

**Course**: Deep Learning (Final Project - Problem 1)
**Project**: Visual Question Answering for Vietnamese Traffic Signs
**Date**: April 25, 2026

---

## 1. Project Objective

Build a VQA system that takes a Vietnamese street image containing traffic signs and a question in Vietnamese as input, and generates a Vietnamese text answer. The project requires implementing and comparing 4 model configurations (A1, A2, B1, B2) with different encoder-decoder architectures.

## 2. Completed Work

### 2.1 Dataset Preparation

| Item | Status | Details |
|------|--------|---------|
| Detection dataset collection | Done | 3,680 images (1920x1080) from Roboflow |
| Class remapping to QCVN 41:2024 | Done | Remapped from 58 raw classes to 42 standardized classes |
| Class taxonomy documentation | Done | Full 93-class taxonomy documented per QCVN 41:2024/BGTVT |
| VQA template generation | Done | 67,144 QA pairs generated from detection annotations |
| VQA question types | Done | 6 types: count, recognition, yes_no, attribute, spatial, reasoning |
| LLM paraphrase augmentation | In Progress | Target: ~280,000 QA pairs via Qwen3.5:9b paraphrase |

### 2.2 Model Implementation

| Component | Status | File |
|-----------|--------|------|
| Co-Attention module | Done | `models/co_attention.py` |
| LSTM Decoder (with cross-attention) | Done | `models/decoder_lstm.py` |
| Transformer Decoder (with positional encoding) | Done | `models/decoder_transformer.py` |
| VQAModelA (A1 + A2) | Done | `models/model_a.py` |
| VQAModelB (B1 zero-shot + B2 LoRA) | Done | `models/model_b.py` |
| VQA Dataset class | Done | `data_utils/dataset.py` |
| Data Collator | Done | `data_utils/collator.py` |

### 2.3 Training Pipeline

| Component | Status | File |
|-----------|--------|------|
| Training config (all hyperparameters) | Done | `train/config.py` |
| Training loop for A1/A2 (2-phase) | Done | `train/train_a.py` |
| Training loop for B2 (LoRA) | Done | `train/train_b.py` |
| Mixed precision (AMP) | Done | Integrated in both training loops |
| Gradient clipping | Done | max_norm=1.0 |
| Checkpoint saving (best + last) | Done | Based on validation loss |

### 2.4 Evaluation & Demo

| Component | Status | File |
|-----------|--------|------|
| Metrics (BLEU, ROUGE-L, BERTScore, VQA Accuracy) | Done | `evaluate/metrics.py` |
| Evaluation script (all 4 configs) | Done | `evaluate/evaluate.py` |
| Gradio demo interface | Done | `demo/app.py` |
| Greedy search generation | Done | Implemented in `model_a.py` |

### 2.5 Utility Scripts

| Script | Status | Purpose |
|--------|--------|---------|
| `scripts/remap_v2.py` | Done | Remap detection class names to QCVN standard |
| `scripts/generate_vqa.py` | Done | Generate VQA QA pairs from YOLO annotations |
| `scripts/check_classes.py` | Done | Analyze class distribution in dataset |

## 3. Architecture Summary

### Approach A: Custom Dual-Encoder

- **Image Encoder**: CLIP ViT-B/16 (frozen) -> 197 patch tokens x 512 dim -> projected to 768 dim
- **Text Encoder**: PhoBERT-base (frozen) -> N tokens x 768 dim
- **Fusion**: 2-layer Co-Attention stack (bidirectional cross-attention + FFN + LayerNorm)
- **Decoder A1**: 2-layer LSTM with cross-attention to encoder memory
- **Decoder A2**: 2-layer Transformer decoder with causal masking and Pre-LN
- **Training**: 2-phase strategy (15 epochs frozen encoders + 15 epochs unfrozen with discriminative lr)
- **Loss**: CrossEntropyLoss (ignore_index=-100 for padding)

### Approach B: Large Vision-Language Model

- **B1**: Qwen-VL-Chat zero-shot (fp16, no training)
- **B2**: Qwen-VL-Chat + LoRA fine-tuning (r=16, alpha=32, bf16, 10 epochs)

## 4. Remaining Work

| Task | Priority | Estimated Effort |
|------|----------|-----------------|
| Complete LLM paraphrase augmentation (~280K QA pairs) | High | Data generation running |
| Run training for A1 configuration | High | ~6 hours GPU time |
| Run training for A2 configuration | High | ~6 hours GPU time |
| Run training for B2 configuration | High | ~3 hours GPU time |
| Run B1 zero-shot evaluation | Medium | ~1 hour |
| Run full evaluation on test set (all 4 configs) | High | After training completes |
| Fill in results table in report | High | After evaluation |
| Record demo video | Medium | After training |
| Implement beam search (optional improvement) | Low | Currently using greedy search |

## 5. Technical Environment

| Item | Specification |
|------|---------------|
| GPU | NVIDIA GeForce RTX 5070 Ti (16GB VRAM GDDR7) |
| OS | Ubuntu Linux |
| CUDA | 13.1, Driver 590.48.01 |
| Python | 3.x (conda env: `d2l`) |
| PyTorch | 2.10.0 |
| Transformers | 4.57.6 |

## 6. Expected Results Table (To Be Filled After Training)

| Metric | A1 (LSTM) | A2 (Transformer) | B1 (Zero-shot) | B2 (LoRA) |
|--------|-----------|-------------------|-----------------|-----------|
| BLEU-1 | -- | -- | -- | -- |
| BLEU-4 | -- | -- | -- | -- |
| ROUGE-L | -- | -- | -- | -- |
| BERTScore-F1 | -- | -- | -- | -- |
| VQA Accuracy | -- | -- | -- | -- |
| Inference (ms/sample) | -- | -- | -- | -- |
| Trainable params | ~45M | ~61M | 0 | ~20M |

## 7. Risk Assessment

| Risk | Mitigation |
|------|------------|
| VQA paraphrase generation taking too long | Can train on template dataset (67K pairs) first, augment later |
| A1/A2 models may underperform on spatial questions | Co-Attention design preserves patch-level spatial information |
| B2 training may run out of VRAM | Using bf16 + gradient accumulation (effective batch=32 with batch_size=8) |
| Qwen-VL download size (~15GB) | Already accounted for in disk space planning |
