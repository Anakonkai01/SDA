# ============================================================
# RAG Pipeline - Hệ thống hỏi đáp Luật Giao thông Việt Nam
# ============================================================
# Cách dùng:
#   1. Chạy build_knowledge_base() 1 lần để tạo vector DB
#   2. Chạy ask() để hỏi đáp
# ============================================================

import os
from pathlib import Path

# ── Thư viện đọc PDF ──────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

# ── Cắt văn bản thành từng đoạn nhỏ ──────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ── Chuyển text thành vector số (embedding) ───────────────
from langchain_huggingface import HuggingFaceEmbeddings
# ── Lưu và tìm kiếm vector ────────────────────────────────
from langchain_community.vectorstores import FAISS

# ── Các thành phần để build RAG chain ─────────────────────
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# BƯỚC 1: ĐỌC VÀ XỬ LÝ PDF
# ============================================================

def load_pdfs(pdf_folder: str) -> list:
    """
    Đọc tất cả file PDF trong thư mục.
    Trả về list các Document (mỗi Document = 1 trang PDF).
    """
    all_docs = []
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    print(f"Tìm thấy {len(pdf_files)} file PDF:")
    for pdf_path in pdf_files:
        print(f"  → Đang đọc: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
    
    print(f"\nTổng cộng: {len(all_docs)} trang\n")
    return all_docs


def chunk_documents(docs: list) -> list:
    """
    Cắt văn bản thành đoạn nhỏ ~512 ký tự, overlap 100 ký tự.
    
    Tại sao cần overlap? Vì nếu câu hỏi liên quan đến ranh giới
    giữa 2 đoạn, overlap đảm bảo context không bị mất.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,        # Độ dài mỗi chunk (ký tự)
        chunk_overlap=100,     # Số ký tự overlap giữa 2 chunk liên tiếp
        separators=[           # Ưu tiên cắt theo thứ tự này
            "\nĐiều ",         # Cắt theo Điều luật trước tiên
            "\nKhoản ",        # Rồi đến Khoản
            "\n\n",            # Rồi paragraph
            "\n",              # Rồi dòng
            ".",               # Cuối cùng mới cắt giữa câu
        ]
    )
    
    chunks = splitter.split_documents(docs)
    print(f"Chia thành {len(chunks)} chunks\n")
    return chunks


# ============================================================
# BƯỚC 2: TẠO EMBEDDING VÀ LƯU VÀO VECTOR DB
# ============================================================

def build_knowledge_base(pdf_folder: str, db_path: str = "faiss_db"):
    """
    Pipeline hoàn chỉnh: PDF → chunks → embeddings → FAISS index
    
    Chỉ cần chạy 1 lần. Sau đó load từ db_path để dùng lại.
    """
    # 1. Đọc PDF
    docs = load_pdfs(pdf_folder)
    
    # 2. Cắt chunks
    chunks = chunk_documents(docs)
    
    # 3. Load embedding model
    # bge-m3: model đa ngôn ngữ, hiểu tiếng Việt rất tốt, miễn phí
    print("Đang load embedding model (lần đầu sẽ tải ~570MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},   # Đổi sang "cpu" nếu không có GPU
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # 4. Tạo FAISS index từ chunks
    print("Đang tạo vector index (có thể mất vài phút)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 5. Lưu lại để dùng sau, không cần build lại
    vectorstore.save_local(db_path)
    print(f"\nĐã lưu Knowledge Base vào '{db_path}/'")
    print("Lần sau chỉ cần load, không cần build lại!\n")
    
    return vectorstore


def load_knowledge_base(db_path: str = "faiss_db"):
    """Load Knowledge Base đã build sẵn."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        db_path, embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Đã load Knowledge Base từ '{db_path}'")
    return vectorstore


# ============================================================
# BƯỚC 3: RAG CHAIN
# ============================================================

# Prompt template - đây là "hướng dẫn" cho LLM
PROMPT_TEMPLATE = """Bạn là trợ lý tư vấn luật giao thông Việt Nam.
Hãy trả lời câu hỏi DỰA TRÊN tài liệu được cung cấp.
Nếu tài liệu không có thông tin, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu."
Trả lời bằng tiếng Việt, ngắn gọn và chính xác.

Tài liệu tham khảo:
{context}

Câu hỏi: {question}

Trả lời:"""


def build_rag_chain(vectorstore, llm):
    """
    Ghép retriever + prompt + LLM thành 1 chain hoàn chỉnh.
    
    Cách hoạt động:
    câu hỏi → retriever tìm top-5 chunk → nhét vào prompt → LLM trả lời
    """
    # Retriever: tìm top-5 chunk giống nhất với câu hỏi
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Lấy top 5 đoạn liên quan nhất
    )
    
    # Format các chunk thành 1 chuỗi text
    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Nguồn: {doc.metadata.get('source', 'N/A')}, trang {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        )
    
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Chain: input → retrieve → format → prompt → LLM → output
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


# ============================================================
# BƯỚC 4: CHẠY 4 CẤU HÌNH (A, B, C, D)
# ============================================================

def run_experiment(question: str, llm_base, llm_finetuned, retriever):
    """
    Chạy câu hỏi qua cả 4 cấu hình để so sánh.
    
    A: LLM gốc, không RAG
    B: LLM gốc, có RAG
    C: LLM fine-tuned, không RAG
    D: LLM fine-tuned, có RAG  ← kỳ vọng tốt nhất
    """
    results = {}
    
    # Lấy context từ retriever (dùng chung cho B và D)
    retrieved_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt_with_context = PROMPT_TEMPLATE.format(
        context=context, question=question
    )
    prompt_without_context = f"Câu hỏi: {question}\nTrả lời:"
    
    # Config A: LLM gốc, không RAG
    results["A"] = llm_base.invoke(prompt_without_context)
    
    # Config B: LLM gốc, có RAG
    results["B"] = llm_base.invoke(prompt_with_context)
    
    # Config C: LLM fine-tuned, không RAG
    results["C"] = llm_finetuned.invoke(prompt_without_context)
    
    # Config D: LLM fine-tuned, có RAG
    results["D"] = llm_finetuned.invoke(prompt_with_context)
    
    return results


# ============================================================
# MAIN - CHẠY THỬ
# ============================================================

if __name__ == "__main__":
    import sys
    
    # ── Chỉnh đường dẫn PDF của bạn ──
    PDF_FOLDER = "./data/laws"   # Thư mục chứa các file PDF
    DB_PATH = "./vector_db"             # Nơi lưu vector DB
    
    # Bước 1: Build knowledge base (chỉ cần 1 lần)
    if not Path(DB_PATH).exists():
        print("=== Lần đầu: đang build Knowledge Base ===")
        vectorstore = build_knowledge_base(PDF_FOLDER, DB_PATH)
    else:
        print("=== Load Knowledge Base đã có sẵn ===")
        vectorstore = load_knowledge_base(DB_PATH)
    
    # Bước 2: Test retriever - kiểm tra xem có tìm đúng không
    print("\n=== TEST RETRIEVER ===")
    test_query = "vượt đèn đỏ bị phạt bao nhiêu tiền?"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(test_query)
    
    print(f"Câu hỏi: '{test_query}'")
    print(f"Tìm được {len(docs)} đoạn liên quan:")
    for i, doc in enumerate(docs, 1):
        print(f"\n[Chunk {i}] Trang {doc.metadata.get('page', '?')}:")
        print(doc.page_content[:200] + "...")
    
    # Bước 3: Thêm LLM vào sau khi fine-tune xong
    # (phần này làm ở bước tiếp theo)
    print("\n✓ Knowledge Base đã sẵn sàng!")
    print("Tiếp theo: sinh QA pairs và fine-tune LLM")