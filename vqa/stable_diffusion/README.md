# Bài 2: Stable Diffusion Text-to-Image

Module này triển khai task Deep Learning độc lập cho Bài 2: sinh ảnh từ mô tả văn bản bằng Stable Diffusion, có demo Gradio và tùy chọn fine-tune LoRA nhỏ từ dataset biển báo giao thông hiện có.

## Cài đặt

Chạy từ thư mục `vqa/`:

```bash
pip install -r requirements.txt
```

Lần chạy đầu tiên sẽ tải model `runwayml/stable-diffusion-v1-5` từ Hugging Face. GPU được khuyến nghị; máy hiện tại có RTX 5070 Ti khoảng 16GB VRAM nên phù hợp cho inference và LoRA nhỏ.

## Chạy demo

```bash
cd vqa
python stable_diffusion/app.py
```

Mở trình duyệt:

```text
http://127.0.0.1:7861
http://100.110.165.40:7861
```

Port `7861` được dùng để không đụng với demo VQA ở port `7860`.

## Prompt mẫu

```text
a realistic Vietnamese street intersection with traffic signs, daytime, high detail
```

```text
a road safety education poster about Vietnamese traffic signs, clean composition
```

```text
a futuristic smart traffic assistant dashboard, Vietnamese city street, traffic signs, cinematic lighting
```

Thiết lập gợi ý:

- Steps: 25
- Guidance scale: 7.5
- Size: 512
- Negative prompt: `blurry, low quality, distorted, unreadable text`
- Seed: đặt một số cố định để tái lập kết quả, hoặc `-1` để random.

## Tạo dataset LoRA từ dữ liệu VQA

Dataset fine-tune được sinh tự động từ metadata có sẵn trong VQA:

- `data/processed/metadata/objects.jsonl`
- `data/processed/images/train/*.jpg`

Mỗi ảnh train được chuyển thành một caption dựa trên class biển báo, nhóm biển, hình dạng, màu sắc và vị trí tương đối.

Smoke dataset:

```bash
cd vqa
python stable_diffusion/build_lora_dataset.py --limit 16 --output stable_diffusion/dataset/metadata.jsonl
```

Dataset chạy qua đêm:

```bash
cd vqa
python stable_diffusion/build_lora_dataset.py --limit 500 --output stable_diffusion/dataset/metadata.jsonl
```

## Fine-tune LoRA qua đêm

Smoke test 2 bước trước:

```bash
cd vqa
python stable_diffusion/train_lora.py \
  --data stable_diffusion/dataset/metadata.jsonl \
  --image-root data/processed \
  --output-dir stable_diffusion/lora_output_smoke \
  --max-train-steps 2 \
  --resolution 512 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --mixed-precision no \
  --learning-rate 1e-5
```

Nếu smoke test ổn, chạy dài hơn:

```bash
cd vqa
python stable_diffusion/train_lora.py \
  --data stable_diffusion/dataset/metadata.jsonl \
  --image-root data/processed \
  --output-dir stable_diffusion/lora_output \
  --max-train-steps 1000 \
  --resolution 512 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --mixed-precision no \
  --learning-rate 1e-5 \
  --save-every 250
```

Trong kiểm thử trên máy hiện tại, `fp16` có thể sinh `nan` ở smoke train, còn `--mixed-precision no --learning-rate 1e-5` chạy ổn định hơn. Nếu muốn tiết kiệm VRAM/tăng tốc, có thể thử lại `fp16`, nhưng nên smoke test trước.

Sau khi có `stable_diffusion/lora_output/`, chạy lại demo. Dropdown model sẽ có lựa chọn `Fine-tuned LoRA`.

## Ghi chú

LoRA không phải điều kiện bắt buộc để nộp. Nếu training chậm hoặc lỗi do tải model/phụ thuộc, bản pretrained inference vẫn là deliverable chính. LoRA chỉ là phần nâng cao để so sánh base model với adapter đã tinh chỉnh trên ảnh biển báo giao thông Việt Nam.
