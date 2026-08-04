# Bài Tập Nhóm — E-commerce Support RAG Chatbot

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| **Dương Văn Kiên*** (Leader) | 2A202601724 | Role 1: Điều phối tiến độ, ghép `supervisor.py`, Task 9 (retrieval pipeline + fallback) | Hoàn thành 100% |
| **Hoàng Thị Hà Huyền** | 2A202601909 | Role 2: Task 1, 3, 4, 5 (thu thập tài liệu, convert markdown, chunking/indexing, semantic search) | Hoàn thành 100% |
| **Lương Hoàng Minh** | 2A202601490 | Role 3: Task 6, 7, 8 (BM25, RRF reranking, PageIndex fallback) | Hoàn thành 100% |
| **Nguyễn Đình Hoàng** | 2A202601436 | Role 4: Task 2, 10, `app.py` (crawl, generation citation, giao diện Streamlit) | Hoàn thành 100% |
| **Trần Tiến Dũng** | 2A202601064 | Role 5: Đánh giá RAGAS, `golden_dataset.json`, `eval_pipeline.py`, `results.md` | Hoàn thành 100% |

---

## Kiến Trúc Hệ Thống

```mermaid
graph TD
    User([User Query]) --> ChatUI(Streamlit Chat UI)
    ChatUI --> RAG_Pipeline
    
    subgraph RAG_Pipeline [Retrieval-Augmented Generation Pipeline]
        Q_Proc(Query Processing) --> Dense(Semantic Search <br> BAAI/bge-m3)
        Q_Proc --> Sparse(Lexical Search <br> BM25)
        
        Dense --> RRF(RRF Reranking)
        Sparse --> RRF
        
        RRF --> CheckScore{Top-1 Score <br> < 0.48?}
        
        CheckScore -- Yes --> PageIndex(PageIndex Fallback <br> Vectorless)
        CheckScore -- No --> Reorder(Document Reordering <br> Lost-in-the-Middle)
        
        PageIndex --> LLM
        Reorder --> LLM(LLM Generation <br> OpenRouter)
    end
    
    LLM --> Response([Response with Citations])
```

---

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ khách hàng liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
