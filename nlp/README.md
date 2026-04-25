# NLP - Vietnamese Traffic Law Question Answering System

A Retrieval-Augmented Generation (RAG) system combined with fine-tuned LLM for answering Vietnamese traffic law questions. This project is part of the **Smart Driving Assistant (SDA)** system and serves as the final project for the Introduction to Natural Language Processing course.

## Overview

The system implements and compares **4 configurations**:

| Config | LLM | RAG | Description |
|--------|-----|-----|-------------|
| **A** | Base (Qwen2.5-3B) | No | Baseline - original LLM without any enhancement |
| **B** | Base (Qwen2.5-3B) | Yes | RAG-only - base LLM with retrieved context |
| **C** | Fine-tuned | No | Fine-tuned LLM without retrieval |
| **D** | Fine-tuned | Yes | Full pipeline - fine-tuned LLM + RAG (best) |

## Architecture

```
                    User Question
                         |
            ┌────────────┼────────────┐
            |                         |
     [Without RAG]              [With RAG]
            |                         |
            |              BAAI/bge-m3 Embedding
            |                         |
            |                  FAISS Vector DB
            |                    (96K+ docs)
            |                         |
            |                  Top-5 Retrieval
            |                         |
            └────────────┬────────────┘
                         |
              Qwen2.5-3B-Instruct
           (Base or LoRA Fine-tuned)
                         |
                      Answer
```

### Pipeline Components

1. **Knowledge Base**: 96,770 Vietnamese legal documents from HuggingFace (`VLSP2025-LegalSML/legal-pretrain`), parsed from HTML, chunked (2000 chars, 200 overlap) with legal-aware separators (Article/Clause/Point)
2. **Embedding**: BAAI/bge-m3 (multilingual, 1024-dim) with FAISS vector store
3. **LLM**: Qwen2.5-3B-Instruct, fine-tuned with QLoRA (4-bit) via Unsloth
4. **QA Generation**: Automated via Ollama (Qwen3:14b local) from retrieved chunks

## Results

Evaluated on 50 manually prepared test questions:

| Config | BLEU | ROUGE-L | BERTScore | Recall@5 |
|--------|------|---------|-----------|----------|
| A (Base, no RAG) | 0.0600 | 0.2698 | 0.7188 | -- |
| B (Base + RAG) | 0.0818 | 0.2657 | 0.6547 | 0.7476 |
| C (Fine-tuned) | 0.1165 | 0.3973 | 0.7777 | -- |
| **D (Fine-tuned + RAG)** | **0.1872** | 0.3842 | 0.6965 | 0.7476 |

**Key findings**:
- Fine-tuning provides the largest improvement (BLEU: +94% from A to C)
- RAG further boosts the fine-tuned model (BLEU: +61% from C to D)
- Config D achieves the best BLEU score (0.1872), 3x better than baseline

## Project Structure

```
nlp/
├── src/
│   ├── build_kb.py              # Build knowledge base (HF dataset -> FAISS)
│   ├── generate_qa.py           # Auto-generate QA pairs via Ollama
│   ├── finetune.py              # Fine-tune Qwen2.5-3B with QLoRA
│   ├── evaluate.py              # Evaluate all 4 configs (A/B/C/D)
│   └── app.py                   # Gradio demo (side-by-side comparison)
├── scripts/
│   └── filter_traffic_laws.py   # Filter traffic-related docs from dataset
├── data/
│   ├── laws/                    # Source PDF documents (traffic laws)
│   ├── qa_pairs/
│   │   ├── qa_dataset.json      # 300 training QA pairs
│   │   └── qa_test.json         # 50 test QA pairs
│   └── traffic_law_docs.json    # Filtered traffic law documents
├── models/
│   └── qwen2.5-3b-lora/
│       ├── checkpoint-38/       # Epoch 2 checkpoint
│       ├── checkpoint-57/       # Epoch 3 checkpoint (final)
│       ├── lora_adapter/        # LoRA adapter weights (~50MB)
│       └── merged/              # Merged full model (~6GB)
├── vector_db/                   # FAISS index + metadata
│   ├── index.faiss
│   └── index.pkl
├── reports/
│   ├── evaluation_results.json  # Quantitative metrics
│   └── predictions_all_configs.json
└── requirements.txt
```

## Setup & Usage

### Prerequisites
- Python 3.11+
- NVIDIA GPU with >= 16GB VRAM (tested on RTX 5070 Ti)
- CUDA 13.1+
- Ollama (for QA generation only)

### Step 1: Build Knowledge Base
```bash
python src/build_kb.py
```
Streams 96K+ legal documents from HuggingFace, parses HTML, chunks text, embeds with bge-m3, and saves to FAISS.

### Step 2: Generate QA Dataset
```bash
# Start Ollama first: ollama serve
# Pull model: ollama pull qwen3:14b
python src/generate_qa.py
```
Generates 300 training + 50 test QA pairs from knowledge base chunks using local LLM.

### Step 3: Fine-tune LLM
```bash
python src/finetune.py
```
Fine-tunes Qwen2.5-3B-Instruct with QLoRA (4-bit quantization) via Unsloth. Training config: 3 epochs, lr=2e-4, effective batch=16.

### Step 4: Evaluate
```bash
python src/evaluate.py
```
Runs all 4 configurations on the test set and outputs BLEU, ROUGE-L, BERTScore, and Recall@5.

### Step 5: Demo
```bash
python src/app.py
```
Launches a Gradio web interface at `http://localhost:7860` showing side-by-side answers from all 4 configurations.

## Fine-tuning Details

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-3B-Instruct |
| Method | QLoRA (4-bit quantization) |
| Framework | Unsloth + TRL SFTTrainer |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine schedule) |
| Batch size | 4 (gradient accumulation 4, effective=16) |
| Max sequence length | 1024 |
| Training loss (final) | ~0.58 |

## Key Design Decisions

- **BAAI/bge-m3 for embedding**: Strong multilingual performance, especially for Vietnamese legal text
- **Legal-aware chunking**: Custom separators (`Dieu`, `Khoan`, `Diem`) preserve Vietnamese legal document structure
- **QLoRA over full fine-tuning**: Enables training a 3B model on 16GB VRAM with minimal quality loss
- **Unsloth**: 2x faster training compared to standard HuggingFace PEFT
- **Ollama for QA generation**: Fully offline, no API costs, reproducible
