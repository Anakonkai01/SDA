"""
build_kb.py - Xây dựng Knowledge Base cho hệ thống hỏi đáp Luật Việt Nam

Pipeline: HuggingFace dataset → parse HTML → cắt chunk → embedding → FAISS

Cách dùng:
    python src/build_kb.py
"""

from pathlib import Path

from bs4 import BeautifulSoup
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CẤU HÌNH
# ============================================================

DATASET_NAME = "VLSP2025-LegalSML/legal-pretrain"
DB_PATH      = "./vector_db"
CHUNK_SIZE   = 2000
CHUNK_OVERLAP = 200


# ============================================================
# BƯỚC 1: ĐỌC DỮ LIỆU TỪ HUGGINGFACE
# ============================================================
# Thay vì đọc PDF (hay lỗi), ta đọc từ dataset có sẵn trên HuggingFace.
# Dataset chứa 96,770 văn bản luật VN ở dạng HTML.
# Ta cần: HTML → text thuần (loại bỏ thẻ HTML, giữ nội dung).

def load_documents() -> list[Document]:
    """
    Stream dataset từ HuggingFace, parse HTML thành text thuần.

    Trả về list các Document — mỗi Document chứa:
      - page_content: nội dung text
      - metadata: thông tin về văn bản (tên, số hiệu, cơ quan ban hành, ngày)
    """
    print("Đang stream dataset từ HuggingFace...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    documents = []
    count = 0

    for row in ds:
        count += 1

        # --- Parse HTML → text ---
        # BeautifulSoup đọc HTML, get_text() lấy phần text, bỏ thẻ HTML
        soup = BeautifulSoup(row["doc_content"], "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        # Bỏ qua văn bản quá ngắn (không có nội dung hữu ích)
        if len(text) < 200:
            continue

        # --- Tạo Document ---
        # Document là đơn vị dữ liệu của LangChain, gồm nội dung + metadata
        metadata = row["metadata"]
        doc = Document(
            page_content=text,
            metadata={
                "source": metadata.get("DocIdentity", ""),
                "title": metadata.get("DocName", ""),
                "organ": metadata.get("OrganName", ""),
                "date": str(metadata.get("IssueDate", ""))[:10],
            },
        )
        documents.append(doc)

        if count % 10000 == 0:
            print(f"  Đã xử lý {count} văn bản, giữ lại {len(documents)}...")

    print(f"\nHoàn tất: {count} văn bản → giữ lại {len(documents)} văn bản\n")
    return documents


# ============================================================
# BƯỚC 2: CẮT CHUNK
# ============================================================
# Mỗi văn bản luật có thể dài hàng trăm trang.
# Cắt thành đoạn nhỏ 2000 ký tự để sau này tìm kiếm chính xác hơn.
# Overlap 200 ký tự để không mất thông tin ở ranh giới.

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Cắt danh sách Document thành các chunk nhỏ."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nĐiều ",      # Ưu tiên cắt theo Điều (đơn vị lớn nhất trong luật)
            "\nKhoản ",      # Rồi theo Khoản
            "\nĐiểm ",      # Rồi theo Điểm
            "\nChương ",     # Rồi theo Chương
            "\nMục ",        # Rồi theo Mục
            "\n\n",          # Rồi đoạn văn
            "\n",            # Rồi dòng
            ". ",            # Cuối cùng mới cắt giữa câu
        ],
    )

    chunks = splitter.split_documents(documents)
    print(f"Đã cắt thành {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")
    return chunks


# ============================================================
# BƯỚC 3 + 4: EMBEDDING VÀ LƯU VÀO FAISS
# ============================================================
# Mỗi chunk được chuyển thành vector (dãy 1024 số) bằng model bge-m3.
# Tất cả vector được lưu vào FAISS để tìm kiếm nhanh sau này.

def build_vector_db(chunks: list[Document], db_path: str = DB_PATH):
    """Embedding tất cả chunks và lưu vào FAISS index."""

    # Load embedding model
    print("Đang load embedding model BAAI/bge-m3...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Tạo FAISS index — bước này tốn thời gian nhất
    # FAISS.from_documents sẽ: embed từng chunk → lưu vector vào index
    print(f"Đang tạo FAISS index từ {len(chunks)} chunks...")
    print("(Có thể mất vài giờ, hãy để máy chạy qua đêm)\n")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Lưu index ra ổ cứng
    vectorstore.save_local(db_path)
    print(f"\nĐã lưu Knowledge Base vào '{db_path}/'")

    return vectorstore


def load_vector_db(db_path: str = DB_PATH):
    """Load Knowledge Base đã build sẵn từ ổ cứng."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        db_path, embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"Đã load Knowledge Base từ '{db_path}'")
    return vectorstore


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Bước 1: Đọc dữ liệu
    documents = load_documents()

    # Bước 2: Cắt chunk
    chunks = chunk_documents(documents)

    # Bước 3+4: Embedding + lưu FAISS
    vectorstore = build_vector_db(chunks)

    # Test thử: tìm kiếm 1 câu hỏi
    print("\n=== TEST RETRIEVER ===")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    test_query = "Công ty không đóng bảo hiểm cho nhân viên bị phạt thế nào?"
    docs = retriever.invoke(test_query)

    print(f"Câu hỏi: '{test_query}'")
    print(f"Tìm được {len(docs)} đoạn liên quan:\n")
    for i, doc in enumerate(docs, 1):
        print(f"[Chunk {i}] Nguồn: {doc.metadata.get('title', '?')}")
        print(doc.page_content[:300])
        print("...\n")
