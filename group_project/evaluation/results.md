# RAG Evaluation Results

## Framework sử dụng

RAGAS 0.1.21 — LLM giám khảo qua OpenRouter (cùng model Task 10), embeddings BAAI/bge-m3 local (chỉ dùng cho answer_relevancy).

---

## Overall Scores

| Metric | hybrid_rerank | hybrid_no_rerank | Δ |
|---|---|---|---|
| faithfulness | 0.810 | 0.869 | -0.059 |
| answer_relevancy | 0.758 | 0.707 | +0.051 |
| context_recall | 0.882 | 0.882 | +0.000 |
| context_precision | 0.944 | 0.944 | -0.000 |
| **Average** | **0.848** | **0.850** | -0.002 |

---

## A/B Comparison Analysis

**hybrid_rerank:**
> Average score: 0.848

**hybrid_no_rerank:**
> Average score: 0.850

**Kết luận:**
> Config `hybrid_no_rerank` (0.850) cho điểm trung bình nhỉnh hơn một chút so với `hybrid_rerank` (0.848). Việc không dùng RRF giúp **Faithfulness** cao hơn đáng kể (0.869 so với 0.810), cho thấy LLM bám sát nội dung tốt hơn khi thứ tự ngữ cảnh không bị xáo trộn. Ngược lại, việc dùng RRF giúp cải thiện **Answer Relevance** (0.758 so với 0.707). Vì Context Recall và Precision bằng nhau, nhóm có thể cân nhắc bỏ RRF để tiết kiệm thời gian xử lý và giảm tỷ lệ hallucination.
---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|---|---|---|---|---|---|---|
| 1 | Tôi có thể yêu cầu trả hàng vì lý do 'đổi ý' không? | 0.00 | 0.00 | 0.00 | 0.70 | **Retrieval** | Chunking cắt mất đoạn chứa từ khóa "đổi ý" hoặc do hệ thống không tìm thấy sự liên kết ngữ nghĩa giữa "đổi ý" và "không còn nhu cầu". |
| 2 | Thời gian bảo hành sản phẩm thông qua Shopee mất khoảng bao lâu? | 1.00 | 0.00 | 1.00 | 1.00 | **Generation** | Mặc dù tìm ra đúng văn bản (Recall=1) và trả lời đúng fact (Faithfulness=1), nhưng cách LLM diễn đạt dài dòng, không trả lời thẳng vào trọng tâm "bao lâu" khiến Ragas chấm Relevance = 0. |
| 3 | Có những cách nào để gửi yêu cầu Trả hàng/Hoàn tiền trên Shopee? | 0.75 | 0.87 | 0.50 | 1.00 | **Retrieval** | Context Recall = 0.5 cho thấy chỉ tìm được một phần tài liệu (có thể do chunk_size cắt ngang đoạn liệt kê các cách). LLM đành trả lời dựa trên 1 nửa thông tin. |

---

## Recommendations

### Cải tiến 1
**Action:** Tối ưu hóa Chunking Strategy (Sử dụng Semantic Chunking hoặc tăng `chunk_overlap`).
**Expected impact:** Giải quyết tình trạng cắt ngang câu (như ở câu hỏi số 3), đảm bảo các danh sách liệt kê cách thức, quy trình được giữ trọn vẹn trong một chunk, từ đó tăng Context Recall.

### Cải tiến 2
**Action:** Áp dụng Query Expansion hoặc HyDE (Hypothetical Document Embeddings).
**Expected impact:** Rất hiệu quả với câu hỏi mang tính khẩu ngữ như "đổi ý" (câu hỏi 1). HyDE sẽ sinh ra bản nháp "không còn nhu cầu sử dụng" để giúp Vector Database dễ dàng match với từ khóa thực tế trong văn bản chính sách.

### Cải tiến 3
**Action:** Cập nhật lại System Prompt cho LLM.
**Expected impact:** Yêu cầu LLM trả lời ngắn gọn, đi thẳng vào vấn đề thay vì liệt kê dông dài toàn bộ quy định (khắc phục lỗi Answer Relevance = 0 ở câu hỏi số 2 dù đã cung cấp đúng tài liệu).
