# VQA Biển Báo Giao Thông VN — Design Document
**Dành cho Claude Code implement**
**Môn: Học Sâu | Nhóm: 3 người**

---

## 1. Tổng quan

Hệ thống VQA (Visual Question Answering) nhận input là ảnh đường phố Việt Nam có biển báo giao thông + câu hỏi tiếng Việt, sinh ra câu trả lời tiếng Việt.

**4 cấu hình cần implement và so sánh:**
- A1: CLIP ViT + PhoBERT + Co-Attention + LSTM decoder
- A2: CLIP ViT + PhoBERT + Co-Attention + Transformer decoder
- B1: Qwen-VL zero-shot
- B2: Qwen-VL fine-tune LoRA

---

## 2. Dataset

### Format
File JSON với cấu trúc:
```json
{
  "metadata": {"total": 280000, "train": 224000, "val": 28000, "test": 28000},
  "train": [
    {
      "image": "data/datasets/remapped_v2/train/images/xxx.png",
      "question": "Biển báo bên trái trong ảnh là gì?",
      "answer": "Biển cấm dừng và đỗ xe",
      "type": "recognition"
    }
  ],
  "val": [...],
  "test": [...]
}
```

### Loại câu hỏi
- `count`: đếm số lượng biển
- `recognition`: nhận dạng loại biển
- `yes_no`: có/không
- `attribute`: thuộc tính biển
- `spatial`: vị trí không gian
- `reasoning`: suy luận kết hợp nhiều biển
- `*_llm`: phiên bản paraphrase bởi LLM

### Ảnh
- Kích thước gốc: 1920x1080
- Resize về: 224x224 (Hướng A), 448x448 (Hướng B)
- Format: RGB

---

## 3. Hướng A — Dual Encoder + Co-Attention

### 3.1 Kiến trúc tổng thể

```
Ảnh (224x224)
      ↓
CLIP ViT-B/16 (frozen)
      ↓ [batch, 197, 512]
Linear(512 → 768)
      ↓ [batch, 197, 768]  — image_tokens

Câu hỏi (text)
      ↓
PhoBERT-base (frozen)
      ↓ [batch, N, 768]    — text_tokens

Co-Attention Module (2 lớp)
  text_enriched  [batch, N, 768]
  image_enriched [batch, 197, 768]

Concat → memory [batch, N+197, 768]

A1: LSTM Decoder       → output [batch, seq, vocab_size]
A2: Transformer Decoder → output [batch, seq, vocab_size]

Loss: Cross-Entropy
```

### 3.2 Co-Attention Module

```python
class CoAttentionLayer(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_to_image = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.image_to_text = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_text_1 = nn.LayerNorm(dim)
        self.norm_text_2 = nn.LayerNorm(dim)
        self.norm_image_1 = nn.LayerNorm(dim)
        self.norm_image_2 = nn.LayerNorm(dim)
        self.ffn_text = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_image = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, text_tokens, image_tokens, text_mask=None):
        # Text attend to Image (residual connection)
        text_out, _ = self.text_to_image(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens
        )
        text_tokens = self.norm_text_1(text_tokens + text_out)
        text_tokens = self.norm_text_2(text_tokens + self.ffn_text(text_tokens))

        # Image attend to Text (residual connection)
        image_out, _ = self.image_to_text(
            query=image_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_mask
        )
        image_tokens = self.norm_image_1(image_tokens + image_out)
        image_tokens = self.norm_image_2(image_tokens + self.ffn_image(image_tokens))

        return text_tokens, image_tokens


class CoAttentionStack(nn.Module):
    def __init__(self, num_layers=2, dim=768, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            CoAttentionLayer(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, text_tokens, image_tokens, text_mask=None):
        for layer in self.layers:
            text_tokens, image_tokens = layer(text_tokens, image_tokens, text_mask)
        return text_tokens, image_tokens
```

### 3.3 Decoder A1 — LSTM

```python
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size=64000, embed_dim=768, hidden_dim=768, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        # Cross-attention với memory (co-attention output)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_input_ids, memory, memory_key_padding_mask=None):
        # tgt_input_ids: [batch, tgt_len]
        # memory: [batch, N+197, 768]
        embedded = self.dropout(self.embedding(tgt_input_ids))
        lstm_out, _ = self.lstm(embedded)  # [batch, tgt_len, hidden_dim]

        # Cross-attention với memory
        attn_out, _ = self.cross_attn(
            query=lstm_out,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )
        out = self.norm(lstm_out + attn_out)
        return self.output_proj(out)  # [batch, tgt_len, vocab_size]
```

### 3.4 Decoder A2 — Transformer

```python
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=64000, d_model=768, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=3072,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN, train ổn định hơn
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_input_ids, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt_input_ids: [batch, tgt_len]
        # memory: [batch, N+197, 768]
        embedded = self.pos_encoding(self.embedding(tgt_input_ids))
        out = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_proj(out)  # [batch, tgt_len, vocab_size]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
```

### 3.5 VQA Model A (đầy đủ)

```python
class VQAModelA(nn.Module):
    def __init__(self, decoder_type="lstm"):
        super().__init__()
        # Image encoder
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.img_proj = nn.Linear(512, 768)

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # Freeze encoders
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Co-Attention
        self.co_attention = CoAttentionStack(num_layers=2)

        # Decoder
        if decoder_type == "lstm":
            self.decoder = LSTMDecoder()
        else:
            self.decoder = TransformerDecoder()

        self.decoder_type = decoder_type

    def encode(self, pixel_values, input_ids, attention_mask):
        # Image features
        img_out = self.image_encoder(pixel_values).last_hidden_state  # [B, 197, 512]
        img_tokens = self.img_proj(img_out)                            # [B, 197, 768]

        # Text features
        txt_out = self.text_encoder(input_ids, attention_mask).last_hidden_state  # [B, N, 768]

        # Co-Attention
        text_mask = (attention_mask == 0)  # padding mask
        text_enriched, image_enriched = self.co_attention(txt_out, img_tokens, text_mask)

        # Concat làm memory
        memory = torch.cat([text_enriched, image_enriched], dim=1)  # [B, N+197, 768]
        return memory

    def forward(self, pixel_values, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask=None):
        memory = self.encode(pixel_values, input_ids, attention_mask)

        if self.decoder_type == "transformer":
            tgt_len = decoder_input_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(memory.device)
            logits = self.decoder(
                decoder_input_ids, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(decoder_attention_mask == 0) if decoder_attention_mask is not None else None
            )
        else:
            logits = self.decoder(decoder_input_ids, memory)

        return logits  # [B, tgt_len, vocab_size]

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, max_length=20, beam_size=4):
        memory = self.encode(pixel_values, input_ids, attention_mask)
        # Implement greedy search hoặc beam search
        # BOS token = tokenizer.bos_token_id
        # EOS token = tokenizer.eos_token_id
        # Return: List[str] — câu trả lời đã decode
        pass
```

### 3.6 Training config Hướng A

```python
config_A = {
    "model": {
        "image_encoder": "openai/clip-vit-base-patch16",
        "text_encoder": "vinai/phobert-base",
        "co_attention_layers": 2,
        "co_attention_heads": 8,
        "dim": 768,
        "dropout": 0.1,
        # LSTM specific
        "lstm_hidden": 768,
        "lstm_layers": 2,
        # Transformer specific
        "transformer_layers": 2,
        "transformer_ffn": 3072,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 3e-4,
        "epochs": 30,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "gradient_clip": 1.0,
        "scheduler": "cosine",
        "mixed_precision": True,
    },
    "data": {
        "max_question_length": 64,
        "max_answer_length": 20,
        "image_size": 224,
        "num_workers": 4,
    },
    "phases": {
        # Phase 1: Freeze encoders (epoch 1-15)
        "phase1_epochs": 15,
        "phase1_lr": 3e-4,
        # Phase 2: Unfreeze encoders với lr nhỏ hơn (epoch 16-30)
        "phase2_epochs": 15,
        "phase2_lr": 3e-5,
    }
}
```

---

## 4. Hướng B — Qwen-VL

### 4.1 B1 — Zero-shot

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    trust_remote_code=True
)
model.eval()

def inference_zero_shot(image_path, question):
    query = tokenizer.from_list_format([
        {"image": image_path},
        {"text": question}
    ])
    with torch.no_grad():
        response, _ = model.chat(tokenizer, query=query, history=None)
    return response

# Evaluate trên toàn bộ test set
def evaluate_zero_shot(test_samples):
    predictions = []
    for sample in tqdm(test_samples):
        pred = inference_zero_shot(sample["image"], sample["question"])
        predictions.append(pred)
    return predictions
```

### 4.2 B2 — LoRA Fine-tune

```python
from peft import LoraConfig, get_peft_model, TaskType

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat",
    torch_dtype=torch.bfloat16,  # bf16 cho RTX 5070 Ti
    device_map="cuda",
    trust_remote_code=True
)

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "w1", "w2"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.3% params trainable
```

### 4.3 Training config Hướng B

```python
config_B = {
    "model": "Qwen/Qwen-VL-Chat",
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["c_attn", "c_proj", "w1", "w2"],
        "lora_dropout": 0.05,
    },
    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 4,  # Effective batch = 32
        "learning_rate": 2e-4,
        "epochs": 10,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_clip": 1.0,
        "scheduler": "cosine",
    },
    "data": {
        "max_length": 256,
        "image_size": 448,
        "num_workers": 4,
    }
}
```

---

## 5. Dataset Class

```python
class VQADataset(Dataset):
    def __init__(self, samples, clip_processor, phobert_tokenizer, max_q_len=64, max_a_len=20, image_size=224):
        self.samples = samples
        self.clip_processor = clip_processor
        self.phobert_tokenizer = phobert_tokenizer
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")

        # Process image
        pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Process question
        q_enc = self.phobert_tokenizer(
            sample["question"],
            max_length=self.max_q_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Process answer (decoder input/output)
        a_enc = self.phobert_tokenizer(
            sample["answer"],
            max_length=self.max_a_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Teacher forcing: decoder_input = BOS + answer[:-1]
        # Labels = answer + EOS (ignore padding = -100)
        decoder_input_ids = torch.cat([
            torch.tensor([self.phobert_tokenizer.bos_token_id]),
            a_enc.input_ids.squeeze()[:-1]
        ])
        labels = a_enc.input_ids.squeeze().clone()
        labels[labels == self.phobert_tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": q_enc.input_ids.squeeze(),
            "attention_mask": q_enc.attention_mask.squeeze(),
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
```

---

## 6. Evaluation Metrics

```python
from evaluate import load
import re

bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

def normalize_answer(ans):
    ans = ans.lower().strip()
    ans = re.sub(r'[^\w\s]', '', ans)
    return ' '.join(ans.split())

def compute_metrics(predictions, references):
    norm_preds = [normalize_answer(p) for p in predictions]
    norm_refs = [normalize_answer(r) for r in references]

    results = {}

    # BLEU
    results["bleu1"] = bleu.compute(
        predictions=norm_preds,
        references=[[r] for r in norm_refs],
        max_order=1
    )["bleu"]
    results["bleu4"] = bleu.compute(
        predictions=norm_preds,
        references=[[r] for r in norm_refs],
        max_order=4
    )["bleu"]

    # ROUGE-L
    results["rouge_l"] = rouge.compute(
        predictions=norm_preds,
        references=norm_refs
    )["rougeL"]

    # BERTScore
    bs = bertscore.compute(
        predictions=norm_preds,
        references=norm_refs,
        lang="vi"
    )
    results["bertscore_f1"] = sum(bs["f1"]) / len(bs["f1"])

    # VQA Accuracy (exact match)
    results["vqa_accuracy"] = sum(
        p == r for p, r in zip(norm_preds, norm_refs)
    ) / len(norm_preds)

    return results
```

---

## 7. File Structure

```
vqa/
├── data/
│   ├── datasets/remapped_v2/
│   └── vqa_final.json
├── models/
│   ├── co_attention.py
│   ├── decoder_lstm.py
│   ├── decoder_transformer.py
│   ├── model_a.py
│   └── model_b.py
├── train/
│   ├── train_a.py
│   ├── train_b.py
│   └── config.py
├── evaluate/
│   ├── evaluate.py
│   └── metrics.py
├── demo/
│   └── app.py          # Gradio demo
└── notebooks/
    └── analysis.ipynb
```

---

## 8. Bảng kết quả (template báo cáo)

| Metric | A1 (LSTM) | A2 (Transformer) | B1 (Zero-shot) | B2 (LoRA) |
|---|---|---|---|---|
| BLEU-1 | | | | |
| BLEU-4 | | | | |
| ROUGE-L | | | | |
| BERTScore-F1 | | | | |
| VQA Accuracy | | | | |
| Inference (ms/sample) | | | | |
| Trainable params | ~45M | ~61M | 0 | ~20M |

---

## 9. Lưu ý khi implement

1. **Teacher forcing khi train** — decoder nhận ground truth tokens làm input
2. **Causal mask cho Transformer decoder** — token không được attend tokens phía sau
3. **Generation khi inference** — dùng beam search (beam=4) hoặc greedy
4. **Padding labels = -100** — CrossEntropyLoss bỏ qua padding
5. **Mixed precision** — dùng `torch.cuda.amp.autocast()` và `GradScaler`
6. **Checkpoint** — save best model theo VQA Accuracy trên val set
7. **Phase training cho Hướng A** — freeze encoders 15 epoch đầu, unfreeze 15 epoch sau