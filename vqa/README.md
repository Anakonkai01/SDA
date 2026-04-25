# VQA - Vietnamese Traffic Sign Visual Question Answering

A Visual Question Answering system for Vietnamese traffic signs. Given a street image containing traffic signs and a Vietnamese question, the system generates a Vietnamese answer. This project is part of the **Smart Driving Assistant (SDA)** system and serves as the final project for the Deep Learning course.

## Overview

The system implements and compares **4 model configurations**:

| Config | Architecture | Description |
|--------|-------------|-------------|
| **A1** | CLIP ViT-B/16 + PhoBERT + Co-Attention + LSTM Decoder | Custom dual-encoder with LSTM generation |
| **A2** | CLIP ViT-B/16 + PhoBERT + Co-Attention + Transformer Decoder | Custom dual-encoder with Transformer generation |
| **B1** | Qwen-VL-Chat (zero-shot) | Large VLM without fine-tuning |
| **B2** | Qwen-VL-Chat + LoRA | Large VLM fine-tuned with LoRA |

## Architecture

### Approach A - Dual Encoder + Co-Attention

```
Image (224x224)              Question (Vietnamese)
      |                              |
CLIP ViT-B/16 (frozen)       PhoBERT-base (frozen)
      |                              |
  [B, 197, 512]                 [B, N, 768]
      |                              |
Linear(512 -> 768)                   |
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

### Approach B - Qwen-VL

- **B1**: Direct zero-shot inference with Qwen-VL-Chat
- **B2**: LoRA fine-tuning (r=16, alpha=32) on traffic sign QA data

## Dataset

### Detection Dataset
- **Source**: Roboflow, remapped to QCVN 41:2024/BGTVT standard
- **Format**: YOLOv8
- **Size**: 3,680 images (1920x1080)
- **Classes**: 42 traffic sign types (Prohibition, Warning, Mandatory, Information, Supplementary)

### VQA Dataset
- **Template**: 67,144 QA pairs generated from detection annotations
- **Target**: ~280,000 QA pairs after LLM paraphrase (in progress)
- **Question types**: `count`, `recognition`, `yes_no`, `attribute`, `spatial`, `reasoning`
- **Format**:
```json
{
  "image": "path/to/image.png",
  "question": "What is the sign on the left?",
  "answer": "No parking sign",
  "type": "recognition"
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
│   ├── generate_vqa.py          # Generate VQA dataset from annotations
│   ├── remap_v2.py              # Remap class names to QCVN standard
│   └── check_classes.py         # Check class distribution
├── data/
│   ├── detection/               # Traffic sign detection dataset
│   └── vqa/                     # VQA question-answer pairs
├── docs/                        # Design documents and requirements
└── requirements.txt
```

## Training

### A1/A2 - Two-Phase Training
```bash
# Train A1 (LSTM decoder)
python train/train_a.py --decoder lstm

# Train A2 (Transformer decoder)
python train/train_a.py --decoder transformer
```

Training uses a 2-phase strategy:
- **Phase 1** (epochs 1-15): Frozen CLIP + PhoBERT encoders, lr=3e-4
- **Phase 2** (epochs 16-30): Unfrozen encoders with discriminative lr (encoders: 3e-5, decoder: 3e-4)

### B2 - LoRA Fine-tuning
```bash
python train/train_b.py
```

LoRA config: r=16, alpha=32, targets=`[c_attn, c_proj, w1, w2]`, effective batch=32

## Evaluation

```bash
# Evaluate single model
python evaluate/evaluate.py --model a1 --checkpoint checkpoints/model_a1/best.pt

# Evaluate all models
python evaluate/evaluate.py --model all
```

**Metrics**: BLEU-1, BLEU-4, ROUGE-L, BERTScore-F1, VQA Accuracy, Inference time

## Demo

```bash
python demo/app.py
```

Launches a Gradio web interface where users can upload a street image, type a Vietnamese question, and select any of the 4 model configurations to get an answer.

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
- **Qwen-VL over BLIP-2**: Better multilingual (especially Vietnamese) support from Asian-language training data
- **LoRA r=16**: Optimal balance between capacity and VRAM usage for domain-specific adaptation
