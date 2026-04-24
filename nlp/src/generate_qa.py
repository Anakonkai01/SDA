# ============================================================
# generate_qa.py - Tự động sinh cặp QA từ Knowledge Base
# ============================================================
# Sử dụng Ollama local (không cần API key)
# Cài đặt: https://ollama.com
# ============================================================

import json
import random
from pathlib import Path

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Cấu hình ──────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b"
DB_PATH      = "./vector_db"
OUTPUT_DIR   = Path("./data/qa_pairs")
N_TRAIN      = 300
N_TEST       = 50
QA_PER_CHUNK = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROMPT
# ============================================================

QA_PROMPT = """Bạn là chuyên gia về luật giao thông Việt Nam.
Dựa trên đoạn văn bản luật dưới đây, hãy tạo {n} cặp câu hỏi - trả lời.

Yêu cầu:
- Câu hỏi phải tự nhiên, như người dùng thực sự sẽ hỏi
- Trả lời phải chính xác, dựa hoàn toàn vào đoạn văn bản
- Đa dạng loại câu hỏi: mức phạt, quy định, điều kiện, định nghĩa
- Trả lời ngắn gọn (1-3 câu), không dài dòng
- Trả lời bằng tiếng Việt

Đoạn văn bản:
\"\"\"
{chunk}
\"\"\"

Trả về JSON theo đúng format sau, KHÔNG thêm gì khác ngoài JSON:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks_from_db(db_path: str) -> list:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        db_path, embeddings,
        allow_dangerous_deserialization=True
    )
    all_docs = list(vectorstore.docstore._dict.values())
    print(f"Tổng số chunks trong DB: {len(all_docs)}")
    return all_docs


def filter_good_chunks(docs: list, min_length: int = 200) -> list:
    good_chunks = []
    for doc in docs:
        text = doc.page_content
        if len(text) < min_length:
            continue
        words = text.split()
        if len(words) < 20:
            continue
        good_chunks.append(doc)
    print(f"Chunks đủ chất lượng: {len(good_chunks)}/{len(docs)}")
    return good_chunks

# ============================================================
# SINH QA
# ============================================================

def generate_qa_from_chunk(chunk_text: str, source: str, n: int = 3) -> list:
    prompt = QA_PROMPT.format(
        n=n,
        chunk=chunk_text[:1500],
    )
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 2048},
        }, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["response"].strip()

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        qa_pairs = json.loads(raw)

        valid = []
        for qa in qa_pairs:
            if "question" in qa and "answer" in qa:
                if len(qa["question"]) > 10 and len(qa["answer"]) > 10:
                    qa["chunk_source"] = source
                    valid.append(qa)
        return valid

    except Exception as e:
        print(f"     Lỗi: {e}")
        return []


def generate_dataset(n_total: int, db_path: str, qa_per_chunk: int = 3) -> list:
    all_docs = load_chunks_from_db(db_path)
    good_chunks = filter_good_chunks(all_docs)
    random.shuffle(good_chunks)

    n_chunks_needed = (n_total // qa_per_chunk) + 20
    chunks_to_process = good_chunks[:n_chunks_needed]

    print(f"\nSẽ xử lý {len(chunks_to_process)} chunks để sinh ~{n_total} cặp QA")
    print(f"Model: {OLLAMA_MODEL} (local)")
    print("Bắt đầu sinh QA...\n")

    all_qa = []
    errors = 0

    for i, doc in enumerate(chunks_to_process):
        if len(all_qa) >= n_total:
            break

        source = (
            f"{Path(doc.metadata.get('source', 'unknown')).name}, "
            f"trang {doc.metadata.get('page', '?')}"
        )

        qa_pairs = generate_qa_from_chunk(doc.page_content, source, qa_per_chunk)

        if qa_pairs:
            all_qa.extend(qa_pairs)
            print(f"[{i+1}/{len(chunks_to_process)}] ✓ +{len(qa_pairs)} cặp | Tổng: {len(all_qa)}/{n_total}")
        else:
            errors += 1
            print(f"[{i+1}/{len(chunks_to_process)}] ✗ Bỏ qua | Lỗi tích lũy: {errors}")

    print(f"\nHoàn thành! Tổng: {len(all_qa)} cặp | Lỗi: {errors} chunks")
    return all_qa[:n_total]

# ============================================================
# LƯU DATASET
# ============================================================

def save_dataset(qa_pairs: list, train_path: str, test_path: str):
    random.shuffle(qa_pairs)

    test_set  = qa_pairs[:N_TEST]
    train_set = qa_pairs[N_TEST:]

    def to_instruction_format(qa: dict) -> dict:
        return {
            "instruction": "Bạn là trợ lý tư vấn luật giao thông Việt Nam. Trả lời chính xác và ngắn gọn.",
            "input": qa["question"],
            "output": qa["answer"],
            "source": qa.get("chunk_source", "")
        }

    train_fmt = [to_instruction_format(qa) for qa in train_set]
    test_fmt  = [to_instruction_format(qa) for qa in test_set]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_fmt, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_fmt, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu:")
    print(f"  Train : {len(train_fmt)} cặp → {train_path}")
    print(f"  Test  : {len(test_fmt)} cặp  → {test_path}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    TRAIN_PATH = str(OUTPUT_DIR / "qa_dataset.json")
    TEST_PATH  = str(OUTPUT_DIR / "qa_test.json")

    if Path(TRAIN_PATH).exists():
        print(f"Dataset đã tồn tại: {TRAIN_PATH}")
        print("Xóa file đó nếu muốn sinh lại.")
    else:
        qa_pairs = generate_dataset(
            n_total=N_TRAIN + N_TEST,
            db_path=DB_PATH,
            qa_per_chunk=QA_PER_CHUNK
        )
        save_dataset(qa_pairs, TRAIN_PATH, TEST_PATH)

    # Preview
    print("\n=== PREVIEW 3 CẶP QA MẪU ===")
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)[:3]

    for i, s in enumerate(samples, 1):
        print(f"\n[{i}] Q: {s['input']}")
        print(f"    A: {s['output']}")
        print(f"    Nguồn: {s['source']}")
