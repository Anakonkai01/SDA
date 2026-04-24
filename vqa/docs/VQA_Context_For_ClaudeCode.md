# VQA Project — Full Context cho Claude Code
**Đọc file này trước khi implement bất cứ thứ gì**

---

## 1. Bối cảnh dự án

### Môn học
- **Môn:** Học Sâu (Deep Learning)
- **Loại:** Dự án cuối kỳ — Bài 1 (7 điểm)
- **Nhóm:** 3 người, deadline ~1 tháng
- **Yêu cầu bắt buộc:** Implement và so sánh đúng 4 cấu hình A1, A2, B1, B2

### Dự án lớn hơn
VQA này là một **subset của dự án SDA (Smart Driving Assistant)** — hệ thống hỗ trợ lái xe thông minh cho Việt Nam. Module VQA sau này sẽ được tích hợp vào SDA để trả lời câu hỏi của tài xế về biển báo giao thông.

---

## 2. Hardware & Environment

### Hardware
- **GPU:** NVIDIA GeForce RTX 5070 Ti — 16GB VRAM GDDR7
- **OS:** Ubuntu (Linux)
- **CUDA:** 13.1, Driver 590.48.01

### Conda Environment
- **Tên env:** `d2l`
- **Python:** 3.x (conda)
- **Packages đã cài:**
  ```
  torch==2.10.0
  torchaudio==2.10.0
  torchvision==0.25.0
  transformers==4.57.6
  pytorch-crf==0.7.2
  ```
- **Packages cần cài thêm:**
  ```bash
  pip install peft accelerate
  pip install open-clip-torch
  pip install evaluate rouge-score bert-score
  pip install gradio
  pip install Pillow tqdm wandb
  ```

---

## 3. Cấu trúc thư mục (đã tổ chức lại)

**Thư mục gốc:** `~/workspace/SDA/vqa/`

```
vqa/
├── data/
│   ├── detection/
│   │   ├── raw/                    # Dataset Roboflow gốc (58 classes, chưa remap)
│   │   └── remapped/               # Dataset đã remap về 42 classes theo QCVN 41:2024
│   │       ├── train/
│   │       │   ├── images/         # Ảnh 1920x1080
│   │       │   └── labels/         # YOLO format annotations
│   │       ├── valid/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       ├── test/
│   │       │   ├── images/
│   │       │   └── labels/
│   │       └── data.yaml           # 42 classes theo QCVN 41:2024
│   └── vqa/
│       ├── vqa_template.json       # 67,144 bộ QA sinh từ template
│       └── vqa_final.json          # ~280,000 bộ QA sau LLM paraphrase (đang generate)
│
├── scripts/                        # Utility scripts (không phải model code)
│   ├── remap_v2.py                 # Remap class names dataset detection
│   ├── generate_vqa.py             # Sinh VQA dataset từ annotation
│   └── check_classes.py            # Kiểm tra phân bố class
│
├── docs/
│   ├── VQA_Design_Document.md      # Kiến trúc kỹ thuật chi tiết
│   ├── VQA_Class_Taxonomy.md       # 42 classes theo QCVN 41:2024
│   └── DU_AN_CUOI_KY_MON_HOC_SAU.docx  # Đề bài gốc
│
├── models/
│   ├── __init__.py
│   ├── co_attention.py             # Co-Attention module
│   ├── decoder_lstm.py             # LSTM decoder
│   ├── decoder_transformer.py      # Transformer decoder + PositionalEncoding
│   ├── model_a.py                  # VQAModelA (A1 + A2)
│   └── model_b.py                  # VQAModelB (B1 + B2 với Qwen-VL)
│
├── data_utils/
│   ├── __init__.py
│   ├── dataset.py                  # VQADataset class
│   └── collator.py                 # DataCollator cho DataLoader
│
├── train/
│   ├── __init__.py
│   ├── train_a.py                  # Training loop Hướng A
│   ├── train_b.py                  # Training loop Hướng B (LoRA)
│   └── config.py                   # Tất cả hyperparameters
│
├── evaluate/
│   ├── __init__.py
│   ├── evaluate.py                 # Chạy evaluation trên test set
│   └── metrics.py                  # BLEU, ROUGE-L, BERTScore, VQA Accuracy
│
├── demo/
│   └── app.py                      # Gradio demo interface
│
├── notebooks/
│   └── analysis.ipynb              # Phân tích và visualize kết quả
│
├── checkpoints/                    # Model checkpoints (gitignore)
│   ├── model_a1/
│   ├── model_a2/
│   └── model_b2/
│
└── requirements.txt
```

---

## 4. Dataset

### 4.1 Detection Dataset

**Đường dẫn:** `data/detection/remapped/`

**Format:** YOLOv8
```
# data.yaml
nc: 42
names: ['DP_het_tat_ca_lenh_cam', 'I_cho_quay_xe', 'I_duong_mot_chieu', ...]
```

**Thống kê:**
- Tổng: 3,680 ảnh (train + valid + test)
- 42 classes theo QCVN 41:2024/BGTVT
- Kích thước ảnh: 1920×1080

**42 classes (đã group theo chức năng):**
```
DP: DP_het_tat_ca_lenh_cam
I:  I_cho_quay_xe, I_duong_mot_chieu
P:  P_cam_coi, P_cam_dung_do_xe, P_cam_nguoc_chieu, P_cam_nguoi_di_bo,
    P_cam_quay_dau, P_cam_re_ca_hai_chieu, P_cam_re_phai, P_cam_re_trai,
    P_cam_vuot, P_cam_xe_khach, P_cam_xe_may, P_cam_xe_oto, P_cam_xe_oto_re,
    P_cam_xe_tai, P_han_che_chieu_cao, P_han_che_trong_tai, P_toc_do_toi_da
R:  R_huong_phai_di, R_khu_dong_dan_cu, R_lan_duong_danh_cho_xe,
    R_vong_chuong_ngai_vat, R_vong_xuyen
S:  S_thuyet_minh
W:  W_cong_truong, W_di_cham, W_doc_nguy_hiem, W_duong_doi, W_duong_giao_nhau,
    W_duong_thu_hep, W_duong_xau, W_giao_nhau_den_tin_hieu,
    W_giao_nhau_duong_nhanh, W_giao_nhau_duong_sat, W_giao_nhau_duong_uu_tien,
    W_ngoat_nguy_hiem, W_nguy_hiem_khac, W_nhieu_ngoat, W_nguoi_di_bo, W_tre_em
```

### 4.2 VQA Dataset

**Đường dẫn:** `data/vqa/vqa_final.json` (đang generate, dùng `vqa_template.json` trước)

**Format:**
```json
{
  "metadata": {
    "total": 280000,
    "train": 224000,
    "val": 28000,
    "test": 28000,
    "model": "qwen3.5:9b"
  },
  "train": [
    {
      "image": "data/detection/remapped/train/images/xxx.png",
      "question": "Biển báo bên trái trong ảnh là gì?",
      "answer": "Biển cấm dừng và đỗ xe",
      "type": "recognition"
    }
  ],
  "val": [...],
  "test": [...]
}
```

**Loại câu hỏi và phân bố:**
```
count:        ~6,700  — đếm số lượng biển
recognition:  ~17,600 — nhận dạng loại biển
yes_no:       ~20,500 — có/không
attribute:    ~11,800 — thuộc tính, hành động cần làm
spatial:      ~10,400 — vị trí không gian
reasoning:    ~tùy    — suy luận kết hợp nhiều biển
*_llm:        ~3x     — phiên bản paraphrase bởi Qwen3.5:9b
```

---

## 5. Các quyết định kiến trúc và lý do

### Tại sao CLIP ViT-B/16 thay vì ResNet50?
ResNet50 compress toàn bộ ảnh thành **1 vector duy nhất** (2048 chiều) — mất hoàn toàn thông tin spatial. Không thể trả lời câu hỏi "biển nằm ở đâu?" hay "có bao nhiêu biển?".

CLIP ViT-B/16 output **197 patch tokens** (14×14 patches + CLS token) — mỗi token đại diện cho 1 vùng 16×16 pixels trong ảnh. Model có thể học được vị trí, số lượng, và quan hệ không gian.

### Tại sao Co-Attention thay vì Concatenate đơn thuần?
Concatenate chỉ "xếp chồng" text và image tokens — model không học được tương tác giữa hai modality.

Co-Attention cho phép:
- Text "hỏi" ảnh: token câu hỏi attend vào vùng ảnh liên quan
- Ảnh "hỏi" text: token ảnh attend vào từ trong câu hỏi liên quan
→ Cả hai được enrich lẫn nhau trước khi vào decoder.

### Tại sao 2 lớp Co-Attention?
1 lớp = text và image "nhìn nhau" 1 lần.
2 lớp = tinh chỉnh thêm lần nữa sau khi đã có context.
3+ lớp = overkill cho task biển báo đơn giản, tăng nguy cơ overfitting.

### Tại sao Linear(512→768) thay vì dùng ViT-L/14?
ViT-B/16 output 512 chiều, PhoBERT output 768 chiều — không match.
Linear projection đơn giản nhất: thêm 1 layer `nn.Linear(512, 768)`.
ViT-L/14 output 768 nhưng model lớn hơn nhiều, tốn VRAM không cần thiết.

### Tại sao Qwen-VL thay vì BLIP-2?
Tiếng Việt: Qwen-VL được train trên dữ liệu đa ngôn ngữ châu Á → tiếng Việt tốt hơn BLIP-2 đáng kể.
Chiến lược: **Direct Vietnamese input** — feed thẳng tiếng Việt, không dịch sang tiếng Anh.

### Tại sao LoRA r=16?
r=8: Ít params, học chậm hơn, có thể underfitting
r=16: Sweet spot — đủ capacity để learn domain biển báo VN, vừa 16GB VRAM
r=32: Tốn VRAM hơn, không cải thiện đáng kể với task này

### Tại sao Text Generation thay vì Classification?
Câu trả lời đa dạng, không thể enumerate hết. Ví dụ "Bên trái, phía trên", "Biển cấm dừng và đỗ xe", "3"... Classification cần vocabulary cố định.

---

## 6. Những thứ KHÔNG làm

- **Không thêm Transformer fusion** ở giữa Co-Attention và Decoder — đã cân nhắc và quyết định giữ đơn giản
- **Không dùng ResNet50** làm image encoder
- **Không dịch** câu hỏi tiếng Việt sang tiếng Anh
- **Không dùng GPT-4o hay API bên ngoài** — phải chạy offline
- **Không thay đổi 4 cấu hình** A1, A2, B1, B2 — đây là yêu cầu bắt buộc của đề bài
- **Không dùng classification head** — phải là text generation

---

## 7. Thứ tự implement

1. `data_utils/dataset.py` — VQADataset
2. `models/co_attention.py` — CoAttentionLayer + CoAttentionStack
3. `models/decoder_lstm.py` — LSTMDecoder
4. `models/decoder_transformer.py` — TransformerDecoder + PositionalEncoding
5. `models/model_a.py` — VQAModelA (gộp A1 + A2)
6. `train/config.py` — Tất cả config
7. `train/train_a.py` — Training loop A1 và A2
8. `models/model_b.py` — VQAModelB (B1 zero-shot + B2 LoRA)
9. `train/train_b.py` — Training loop B2
10. `evaluate/metrics.py` + `evaluate/evaluate.py`
11. `demo/app.py` — Gradio demo

---

## 8. Tham chiếu

- **Design Document chi tiết:** `docs/VQA_Design_Document.md`
- **Class taxonomy:** `docs/VQA_Class_Taxonomy.md`
- **Đề bài gốc:** `docs/DU_AN_CUOI_KY_MON_HOC_SAU.docx`
- **PhoBERT:** `vinai/phobert-base` trên HuggingFace
- **CLIP:** `openai/clip-vit-base-patch16` trên HuggingFace
- **Qwen-VL:** `Qwen/Qwen-VL-Chat` trên HuggingFace