"""
Task 4 — Chunking & Indexing vào Vector Store.

Lựa chọn đã chốt cho bài lab này:

    Chunking: RecursiveCharacterTextSplitter (langchain-text-splitters)
        - CHUNK_SIZE=800 / CHUNK_OVERLAP=100 theo đúng tiêu chí Checkpoint 2.
        - 800 ký tự đủ giữ trọn 1-2 đoạn quy định liền mạch mà không làm loãng
          ngữ cảnh khi nhét cho LLM; 100 ký tự overlap tránh cắt đôi câu văn
          quan trọng ngay ranh giới giữa 2 chunk (ví dụ điều khoản đang liệt kê
          dở danh sách gạch đầu dòng).
        - Recursive splitter ưu tiên tách theo đoạn ("\n\n") rồi mới tới dòng/
          câu/ký tự, nên tôn trọng cấu trúc Markdown của Task 3 hơn so với cắt
          cứng theo số ký tự.

    Embedding: BAAI/bge-m3 (1024 chiều)
        - Multilingual, tối ưu cho cả tiếng Việt lẫn tiếng Anh — phù hợp vì
          toàn bộ corpus là tiếng Việt (Shopee help center).

    Vector store: ChromaDB (PersistentClient, local, không cần Docker)
        - Dùng cosine similarity (hnsw:space="cosine") để tương thích thang đo
          [0,1] mà Task 5/9 dùng làm ngưỡng fallback (score < 0.48).

Metadata `customer_role` (K4 Variant, kế thừa Lab 07): mỗi chunk được gắn nhãn
buyer/seller/both dựa trên từ khoá xuất hiện trong nội dung tài liệu gốc, để
Task 5/9 có thể lọc theo đối tượng áp dụng chính sách.

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800        # Đủ giữ trọn 1-2 đoạn quy định liền mạch cho LLM
CHUNK_OVERLAP = 100     # Tránh cắt đôi câu quan trọng ở ranh giới 2 chunk
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

EMBEDDING_MODEL = "BAAI/bge-m3"  # Multilingual, tốt cho tiếng Việt lẫn tiếng Anh
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "ecommerce_support_docs"

# Từ khoá đơn giản để suy luận customer_role từ nội dung tài liệu gốc.
_SELLER_KEYWORDS = ("người bán", "nhà bán", "gian hàng", "shop", "seller")
_BUYER_KEYWORDS = ("người mua", "khách hàng", "buyer")


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def _infer_customer_role(content: str) -> str:
    """Suy luận customer_role (buyer/seller/both) từ nội dung tài liệu."""
    text = content.lower()
    has_seller = any(kw in text for kw in _SELLER_KEYWORDS)
    has_buyer = any(kw in text for kw in _BUYER_KEYWORDS)

    if has_seller and has_buyer:
        return "both"
    if has_seller:
        return "seller"
    # Mặc định "buyer": toàn bộ corpus là nội dung trung tâm trợ giúp khách mua hàng.
    return "buyer"


def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, 'customer_role': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
                "customer_role": _infer_customer_role(content),
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


_embedding_model = None


def get_embedding_model():
    """Load (và cache) SentenceTransformer model dùng chung cho index + query."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def get_collection():
    """Mở (hoặc tạo) ChromaDB collection dùng chung cho index + query."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào ChromaDB (upsert theo id ổn định source_chunk_index)."""
    collection = get_collection()

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
