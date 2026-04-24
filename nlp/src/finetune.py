# ============================================================
# finetune.py - Fine-tune Qwen2.5-3B với LoRA
# ============================================================
# pip install unsloth
# Chạy: python src/finetune.py
# ============================================================

import json
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# ============================================================
# CẤU HÌNH
# ============================================================

MODEL_NAME   = "unsloth/Qwen2.5-3B-Instruct"   # Tự tải từ HuggingFace
OUTPUT_DIR   = "./models/qwen2.5-3b-lora"
DATA_PATH    = "./data/qa_pairs/qa_dataset.json"

# LoRA config — không cần chỉnh, đây là giá trị tối ưu cho 3B
LORA_R           = 16    # Rank của adapter (càng cao càng nhiều tham số học)
LORA_ALPHA       = 32    # Scaling factor (thường = 2 * r)
LORA_DROPOUT     = 0.05

# Training config
MAX_SEQ_LENGTH   = 1024
BATCH_SIZE       = 4     # Tăng lên 8 nếu VRAM còn dư
GRAD_ACCUM       = 4     # Effective batch = 4 * 4 = 16
EPOCHS           = 3
LEARNING_RATE    = 2e-4
WARMUP_RATIO     = 0.1

# ============================================================
# PROMPT FORMAT
# ============================================================
# Qwen2.5 dùng ChatML format — đây là chuẩn của model này

SYSTEM_PROMPT = "Bạn là trợ lý tư vấn luật giao thông Việt Nam. Trả lời chính xác và ngắn gọn dựa trên luật hiện hành."

def format_prompt(sample: dict) -> str:
    """
    Chuyển 1 cặp QA thành ChatML format để train.
    
    Format:
    <|im_start|>system
    ...
    <|im_end|>
    <|im_start|>user
    câu hỏi
    <|im_end|>
    <|im_start|>assistant
    câu trả lời
    <|im_end|>
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
    )


# ============================================================
# LOAD DỮ LIỆU
# ============================================================

def load_dataset(data_path: str) -> Dataset:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training samples")
    
    # Chuyển thành HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    # Thêm cột "text" chứa formatted prompt
    dataset = dataset.map(
        lambda x: {"text": format_prompt(x)},
        remove_columns=dataset.column_names
    )
    
    print(f"Sample prompt:\n{dataset[0]['text'][:300]}...\n")
    return dataset


# ============================================================
# LOAD MODEL VỚI UNSLOTH
# ============================================================

def load_model():
    print(f"Loading {MODEL_NAME} với 4-bit quantization...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,       # QLoRA: quantize 4-bit để tiết kiệm VRAM
        dtype=None,              # Tự detect (bfloat16 trên 5070 Ti)
    )
    
    # Thêm LoRA adapters vào model
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[         # Các layer sẽ được fine-tune
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Tiết kiệm thêm VRAM
        random_state=42,
    )
    
    # In thống kê tham số
    model.print_trainable_parameters()
    
    return model, tokenizer


# ============================================================
# TRAIN
# ============================================================

def train(model, tokenizer, dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            
            # Batch & gradient
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            
            # Learning rate schedule
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type="cosine",
            
            # Precision
            fp16=False,
            bf16=True,           # RTX 5070 Ti hỗ trợ bf16
            
            # Logging & saving
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,  # Chỉ giữ 2 checkpoint gần nhất
            
            # Dataset
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,
            
            # Seed
            seed=42,
        ),
    )
    
    print("\n=== BẮT ĐẦU TRAINING ===")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Epochs: {EPOCHS}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    trainer_stats = trainer.train()
    
    print(f"\n=== TRAINING XONG ===")
    print(f"Thời gian: {trainer_stats.metrics['train_runtime']:.0f}s "
          f"({trainer_stats.metrics['train_runtime']/60:.1f} phút)")
    print(f"Loss cuối: {trainer_stats.metrics['train_loss']:.4f}")
    
    return trainer


# ============================================================
# LƯU MODEL
# ============================================================

def save_model(model, tokenizer):
    # Lưu LoRA adapter (nhỏ, ~50MB)
    lora_path = OUTPUT_DIR + "/lora_adapter"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\nLoRA adapter đã lưu: {lora_path}")
    
    # Merge LoRA vào base model và lưu (lớn hơn, ~6GB, dùng để inference)
    print("Đang merge LoRA vào base model...")
    merged_path = OUTPUT_DIR + "/merged"
    model.save_pretrained_merged(
        merged_path, tokenizer,
        save_method="merged_16bit"
    )
    print(f"Merged model đã lưu: {merged_path}")


# ============================================================
# TEST INFERENCE SAU TRAINING
# ============================================================

def test_inference(model, tokenizer):
    print("\n=== TEST INFERENCE ===")
    
    FastLanguageModel.for_inference(model)  # Bật inference mode
    
    test_questions = [
        "Vượt đèn đỏ bị phạt bao nhiêu tiền?",
        "Tốc độ tối đa trong khu dân cư là bao nhiêu km/h?",
        "Uống rượu bia khi lái xe bị phạt thế nào?",
    ]
    
    for q in test_questions:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        print(f"\nQ: {q}")
        print(f"A: {response.strip()}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    dataset = load_dataset(DATA_PATH)
    
    # 2. Load model
    model, tokenizer = load_model()
    
    # 3. Train
    train(model, tokenizer, dataset)
    
    # 4. Lưu
    save_model(model, tokenizer)
    
    # 5. Test thử
    test_inference(model, tokenizer)
    
    print("\n✓ Fine-tuning hoàn thành!")
    print("Bước tiếp theo: chạy evaluate.py để so sánh 4 cấu hình A/B/C/D")