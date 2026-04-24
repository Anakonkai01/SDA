# ============================================================
# app.py - Gradio Demo: So sánh 4 cấu hình RAG
# ============================================================
# pip install gradio
# Chạy: python src/app.py
# ============================================================

import torch
import gradio as gr
from unsloth import FastLanguageModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# CẤU HÌNH
# ============================================================

BASE_MODEL_NAME  = "unsloth/Qwen2.5-3B-Instruct"
FINETUNED_PATH   = "./models/qwen2.5-3b-lora/merged"
DB_PATH          = "./vector_db"
MAX_SEQ_LENGTH   = 1024
TOP_K            = 5

SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn luật giao thông Việt Nam. "
    "Trả lời chính xác và ngắn gọn dựa trên luật hiện hành."
)

# ============================================================
# LOAD TẤT CẢ KHI KHỞI ĐỘNG
# ============================================================

print("Đang load models và retriever...")

# Retriever
_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
_vectorstore = FAISS.load_local(
    DB_PATH, _embeddings,
    allow_dangerous_deserialization=True
)
retriever = _vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# Base model
print("Loading base model...")
model_base, tok_base = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model_base)

# Fine-tuned model
print("Loading fine-tuned model...")
model_ft, tok_ft = FastLanguageModel.from_pretrained(
    model_name=FINETUNED_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model_ft)

print("Sẵn sàng!")

# ============================================================
# INFERENCE
# ============================================================

def generate(model, tokenizer, question: str, context: str = None) -> str:
    if context:
        user_msg = f"Dựa vào tài liệu sau:\n{context}\n\nCâu hỏi: {question}"
    else:
        user_msg = question

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()


def get_context(question: str) -> tuple[str, str]:
    """Trả về (context_for_prompt, sources_display)."""
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    sources = "\n".join([
        f"• {d.metadata.get('source', '').split('/')[-1]}, "
        f"trang {d.metadata.get('page', '?')}"
        for d in docs
    ])
    return context, sources


# ============================================================
# HÀM CHÍNH GỌI TỪ GRADIO
# ============================================================

def run_all_configs(question: str):
    if not question.strip():
        empty = "Vui lòng nhập câu hỏi."
        return empty, empty, empty, empty, ""

    context, sources = get_context(question)

    ans_a = generate(model_base, tok_base, question, context=None)
    ans_b = generate(model_base, tok_base, question, context=context)
    ans_c = generate(model_ft,   tok_ft,   question, context=None)
    ans_d = generate(model_ft,   tok_ft,   question, context=context)

    return ans_a, ans_b, ans_c, ans_d, sources


# ============================================================
# GIAO DIỆN GRADIO
# ============================================================

EXAMPLES = [
    "Vượt đèn đỏ bị phạt bao nhiêu tiền?",
    "Tốc độ tối đa trong khu dân cư là bao nhiêu?",
    "Uống rượu bia lái xe bị xử lý thế nào?",
    "Xe máy không đội mũ bảo hiểm phạt bao nhiêu?",
    "Điều kiện để được cấp giấy phép lái xe hạng B2?",
    "Đi ngược chiều bị phạt bao nhiêu?",
]

with gr.Blocks(title="Hỏi đáp Luật Giao thông VN", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Hệ thống Hỏi đáp Luật Giao thông Việt Nam
    So sánh 4 cấu hình: **A** (base) · **B** (base + RAG) · **C** (fine-tuned) · **D** (fine-tuned + RAG)
    """)

    with gr.Row():
        question_box = gr.Textbox(
            label="Câu hỏi",
            placeholder="VD: Vượt đèn đỏ bị phạt bao nhiêu tiền?",
            lines=2,
            scale=4,
        )
        submit_btn = gr.Button("Hỏi", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=question_box, label="Câu hỏi mẫu")

    with gr.Row():
        out_a = gr.Textbox(label="A — LLM gốc, không RAG",        lines=6)
        out_b = gr.Textbox(label="B — LLM gốc + RAG",             lines=6)
    with gr.Row():
        out_c = gr.Textbox(label="C — Fine-tuned, không RAG",      lines=6)
        out_d = gr.Textbox(
            label="D — Fine-tuned + RAG  ★",
            lines=6,
            elem_classes=["best-config"]
        )

    sources_box = gr.Textbox(
        label=f"Nguồn tài liệu retrieved (top {TOP_K} chunks)",
        lines=5,
        interactive=False,
    )

    submit_btn.click(
        fn=run_all_configs,
        inputs=question_box,
        outputs=[out_a, out_b, out_c, out_d, sources_box],
    )
    question_box.submit(
        fn=run_all_configs,
        inputs=question_box,
        outputs=[out_a, out_b, out_c, out_d, sources_box],
    )

    gr.Markdown("""
    ---
    **Metrics tự động (trên 50 câu test):**

    | Config | BLEU | ROUGE-L | BERTScore | Recall@5 |
    |--------|------|---------|-----------|----------|
    | A | 0.0600 | 0.2698 | 0.7188 | — |
    | B | 0.0818 | 0.2657 | 0.6547 | 0.7476 |
    | C | 0.1165 | 0.3973 | 0.7777 | — |
    | D | **0.1872** | 0.3842 | 0.6965 | 0.7476 |
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,       # Đổi True nếu muốn public URL để quay video demo
    )