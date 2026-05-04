# Đề bài — Học Sâu (7đ + 3đ)

## Bài 1 (7đ): VQA tiếng Việt

**Input**: ảnh + câu hỏi tiếng Việt → **Output**: câu trả lời ngắn (≤10 từ)

### 1. Dữ liệu
- Domain tự chọn → đang xài biển báo giao thông
- Train: ≥2000 bộ (≥200 ảnh, mỗi ảnh ≥3 câu hỏi)
- Test: ≥50 bộ thủ công, ảnh ko trùng train
- Câu hỏi đa dạng: yes/no, đếm, nhận dạng, thuộc tính, không gian
- Chia 80/10/10
- Khuyến khích augmentation (ảnh: flip/rotate/crop; text: paraphrase/back-translation)

### 2. Mô hình (bắt buộc cả 2 hướng)

**Hướng A — Kiến trúc riêng**:
- Image encoder: CNN pretrained (ResNet/VGG/EfficientNet) hoặc ViT
- Text encoder: LSTM/BiLSTM hoặc PhoBERT
- Fusion: concat/element-wise/co-attention
- **Bắt buộc**: so sánh LSTM decoder vs Transformer decoder (giữ nguyên image + text encoder)

**Hướng B — Multimodal pretrained**:
- Fine-tune BLIP/BLIP-2/ViLT/LLaVA/Qwen-VL/PaliGemma (LoRA/PEFT nếu cần)
- Nêu rõ xử lý tiếng Việt (dịch hay dùng trực tiếp)

### 3. Đánh giá (phân tích riêng 1 chương)
- VQA Accuracy (exact match / soft accuracy theo VQA v2)
- BLEU, ROUGE-L, METEOR
- BERTScore (ngữ nghĩa)
- LLM-as-a-judge

### 4. Thực nghiệm — 4 cấu hình bắt buộc

| Config | Mô tả |
|--------|-------|
| A1 | Hướng A + LSTM decoder |
| A2 | Hướng A + Transformer decoder |
| B1 | Hướng B zero-shot |
| B2 | Hướng B fine-tuned |

So sánh A1 vs A2 → rõ ảnh hưởng decoder LSTM vs Transformer.

### 5. Demo
Khuyến khích GUI (Gradio)

### 6. Nâng cao (1đ)
- RL: PPO (reward = VQA Acc/BERTScore), DPO, hoặc RLHF
- Preference data ≥100 cặp
- So sánh RL vs SFT (metric tự động + human eval)
- Hoặc kỹ thuật khác

### Sản phẩm nộp
- GitHub (README chi tiết)
- Báo cáo 15-20 trang (có chương đánh giá)
- Slide + video demo 3-5 phút
- Dataset + checkpoint (HuggingFace Hub + Drive)

---

## Bài 2 (3đ): Đề xuất task DL khác
- Trình bày bài toán, giải pháp, code, demo
- Phân tích tại sao chọn giải pháp đó
- Nộp như Bài 1
