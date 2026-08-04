"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    from pageindex.client import PageIndexClient
    import json
    from fpdf import FPDF
    import shutil

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Tạo thư mục chứa file PDF tạm thời
    tmp_pdf_dir = Path(__file__).parent.parent / "data" / "_tmp_pdf"
    tmp_pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo mapping để lưu lại doc_id
    doc_mapping = {}

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        pdf_path = tmp_pdf_dir / f"{md_file.stem}.pdf"
        
        # Convert Markdown -> PDF đơn giản bằng fpdf2
        # Sử dụng encoding latin-1 'replace' tạm thời nếu thiếu font unicode
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("sysfont", "", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", uni=True)
        try:
            pdf.set_font("sysfont", size=12)
        except Exception:
            pdf.set_font("helvetica", size=12) # Fallback
            content = content.encode('latin-1', 'replace').decode('latin-1')

        pdf.multi_cell(0, 7, txt=content)
        pdf.output(str(pdf_path))

        print(f"Uploading {pdf_path.name}...")
        try:
            resp = client.submit_document(str(pdf_path))
            doc_id = resp.get("doc_id") or resp.get("id")
            print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
            doc_mapping[md_file.name] = doc_id
        except Exception as e:
            print(f"  ❌ Upload failed for {md_file.name}: {e}")
            
    # Lưu mapping
    mapping_path = Path(__file__).parent.parent / "pageindex_doc_ids.json"
    mapping_path.write_text(json.dumps(doc_mapping, ensure_ascii=False, indent=2))
    print(f"\nĐã lưu danh sách doc_id vào {mapping_path.name}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.
    """
    from pageindex.client import PageIndexClient
    import time
    import json
    
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    mapping_path = Path(__file__).parent.parent / "pageindex_doc_ids.json"
    
    if not mapping_path.exists():
        print("⚠ Không tìm thấy pageindex_doc_ids.json. Vui lòng chạy upload_documents() trước.")
        return []
        
    doc_mapping = json.loads(mapping_path.read_text())
    doc_ids = list(doc_mapping.values())
    if not doc_ids:
        return []

    results = []
    
    # PageIndex submit_query thường query trên từng tài liệu. 
    # Ta sẽ demo query trên tài liệu đầu tiên hoặc submit query global (tuỳ sdk hỗ trợ)
    # Ở đây thử gửi query lên tài liệu đầu tiên làm ví dụ (nếu API bắt buộc truyền doc_id)
    doc_id = doc_ids[0] 
    try:
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp.get("retrieval_id") or resp.get("id")
        
        # Poll kết quả
        for _ in range(10):
            time.sleep(3)
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") == "completed":
                break
        
        # Parse kết quả
        for node in retrieval.get("retrieved_nodes", [])[:2]:
            for group in node.get("relevant_contents", []):
                for item in group:
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": 0.5, # Default score
                        "metadata": {"section": item.get("section_title", "Unknown")},
                        "source": "pageindex",
                    })
    except Exception as e:
        print(f"Lỗi khi truy vấn PageIndex: {e}")
        
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
