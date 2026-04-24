# ============================================================
# evaluate.py - So sánh 4 cấu hình A / B / C / D
# ============================================================
# A: LLM gốc,       không RAG
# B: LLM gốc,       có RAG
# C: LLM fine-tuned, không RAG
# D: LLM fine-tuned, có RAG   ← kỳ vọng tốt nhất
#
# Metrics: BLEU, ROUGE-L, BERTScore, Recall@5
# ============================================================

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from unsloth import FastLanguageModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ============================================================
# CẤU HÌNH
# ============================================================

BASE_MODEL_NAME    = "unsloth/Qwen2.5-3B-Instruct"
FINETUNED_PATH     = "./models/qwen2.5-3b-lora/merged"
DB_PATH            = "./vector_db"
TEST_DATA_PATH     = "./data/qa_pairs/qa_test.json"
OUTPUT_DIR         = Path("./reports")
MAX_SEQ_LENGTH     = 1024
TOP_K              = 5   # Số chunk retrieve

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = "Bạn là trợ lý tư vấn luật giao thông Việt Nam. Trả lời chính xác và ngắn gọn dựa trên luật hiện hành."

# ============================================================
# LOAD COMPONENTS
# ============================================================

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        DB_PATH, embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def load_llm(model_path: str, is_finetuned: bool = False):
    """Load model với unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, context: str = None) -> str:
    """Sinh câu trả lời từ model."""
    if context:
        user_content = f"Dựa vào tài liệu sau:\n{context}\n\nCâu hỏi: {question}"
    else:
        user_content = question

    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
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
    return response.strip()

# ============================================================
# METRICS
# ============================================================

def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    smoother    = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoother)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_recall_at_k(question: str, reference: str, retriever) -> float:
    """
    Recall@5: kiểm tra xem trong top-5 chunk retrieve được,
    có chunk nào chứa nội dung liên quan đến câu trả lời đúng không.
    Heuristic: nếu ít nhất 3 từ của reference xuất hiện trong chunks.
    """
    docs = retriever.invoke(question)
    context = " ".join([d.page_content.lower() for d in docs])
    ref_words = [w for w in reference.lower().split() if len(w) > 3]
    if not ref_words:
        return 0.0
    hits = sum(1 for w in ref_words if w in context)
    return hits / len(ref_words)

# ============================================================
# CHẠY ĐÁNH GIÁ
# ============================================================

def evaluate_config(config_name: str, model, tokenizer,
                    test_data: list, retriever=None) -> dict:
    """Chạy đánh giá 1 config trên toàn bộ test set."""
    print(f"\n{'='*50}")
    print(f"Đánh giá Config {config_name}...")
    print(f"{'='*50}")

    predictions = []
    references  = []
    recall_scores = []

    for item in tqdm(test_data, desc=f"Config {config_name}"):
        question  = item["input"]
        reference = item["output"]

        # Lấy context nếu dùng RAG
        context = None
        if retriever is not None:
            docs = retriever.invoke(question)
            context = "\n\n".join([d.page_content for d in docs])
            recall = compute_recall_at_k(question, reference, retriever)
            recall_scores.append(recall)

        # Sinh câu trả lời
        prediction = generate_answer(model, tokenizer, question, context)
        predictions.append(prediction)
        references.append(reference)

    # Tính BLEU và ROUGE-L
    bleu_scores   = [compute_bleu(r, p) for r, p in zip(references, predictions)]
    rouge_scores  = [compute_rouge_l(r, p) for r, p in zip(references, predictions)]

    # Tính BERTScore (batch để nhanh hơn)
    print(f"  Đang tính BERTScore...")
    _, _, F1 = bert_score(
        predictions, references,
        lang="vi",
        model_type="bert-base-multilingual-cased",
        verbose=False
    )
    bert_scores = F1.tolist()

    results = {
        "config": config_name,
        "n_samples": len(test_data),
        "bleu":      round(np.mean(bleu_scores), 4),
        "rouge_l":   round(np.mean(rouge_scores), 4),
        "bert_score":round(np.mean(bert_scores), 4),
        "recall_at_5": round(np.mean(recall_scores), 4) if recall_scores else None,
        "predictions": [
            {"question": q, "reference": r, "prediction": p}
            for q, r, p in zip(
                [d["input"] for d in test_data],
                references, predictions
            )
        ]
    }

    print(f"  BLEU:        {results['bleu']:.4f}")
    print(f"  ROUGE-L:     {results['rouge_l']:.4f}")
    print(f"  BERTScore:   {results['bert_score']:.4f}")
    if results["recall_at_5"] is not None:
        print(f"  Recall@5:    {results['recall_at_5']:.4f}")

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import nltk
    nltk.download("punkt", quiet=True)

    # Load test data
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test set: {len(test_data)} câu hỏi")

    # Load retriever
    print("\nLoading retriever...")
    retriever = load_retriever()

    all_results = {}

    # ── Config A: LLM gốc, không RAG ──────────────────────
    print("\nLoading base model...")
    model_base, tok_base = load_llm(BASE_MODEL_NAME, is_finetuned=False)
    all_results["A"] = evaluate_config("A", model_base, tok_base, test_data, retriever=None)
    all_results["B"] = evaluate_config("B", model_base, tok_base, test_data, retriever=retriever)
    del model_base, tok_base
    torch.cuda.empty_cache()

    # ── Config C & D: Fine-tuned LLM ──────────────────────
    print("\nLoading fine-tuned model...")
    model_ft, tok_ft = load_llm(FINETUNED_PATH, is_finetuned=True)
    all_results["C"] = evaluate_config("C", model_ft, tok_ft, test_data, retriever=None)
    all_results["D"] = evaluate_config("D", model_ft, tok_ft, test_data, retriever=retriever)
    del model_ft, tok_ft
    torch.cuda.empty_cache()

    # ── In bảng tổng kết ──────────────────────────────────
    print("\n" + "="*60)
    print("KẾT QUẢ SO SÁNH 4 CẤU HÌNH")
    print("="*60)
    print(f"{'Config':<8} {'BLEU':>8} {'ROUGE-L':>10} {'BERTScore':>12} {'Recall@5':>10}")
    print("-"*60)
    for cfg in ["A", "B", "C", "D"]:
        r = all_results[cfg]
        recall = f"{r['recall_at_5']:.4f}" if r["recall_at_5"] else "   N/A"
        print(f"  {cfg}      {r['bleu']:>8.4f} {r['rouge_l']:>10.4f} {r['bert_score']:>12.4f} {recall:>10}")
    print("="*60)
    print("Ghi chú: A=base, B=base+RAG, C=fine-tuned, D=fine-tuned+RAG")

    # Lưu kết quả chi tiết
    output_path = OUTPUT_DIR / "evaluation_results.json"
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                 for k, v in all_results.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nKết quả lưu tại: {output_path}")

    # Lưu predictions để human eval
    pred_path = OUTPUT_DIR / "predictions_all_configs.json"
    pred_data = {k: v["predictions"][:10] for k, v in all_results.items()}
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=2)
    print(f"Predictions mẫu lưu tại: {pred_path}")