"""
RAG Chatbot — E-commerce Support (Starter Template)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Shopee Trợ Giúp — RAG Chatbot",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🛍️ Shopee Trợ Giúp")
    st.caption("Trợ lý hỏi đáp về chính sách thương mại điện tử và hỗ trợ khách hàng (đổi trả, thanh toán, bảo mật, người bán)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để đổi phương thức thanh toán đơn hàng?",
        "Quy định về đăng bán sản phẩm cho người bán?",
        "Cách mua hàng trên Shopee của quốc gia khác?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5)

    st.divider()
    st.subheader("🔧 Cấu hình pipeline")
    st.caption("Tham số thật đang dùng trong code — đọc trực tiếp từ src/, không phải giá trị mẫu.")

    try:
        from src.task4_chunking_indexing import (
            CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_METHOD,
            EMBEDDING_MODEL, EMBEDDING_DIM, VECTOR_STORE,
        )
        from src.task9_retrieval_pipeline import SCORE_THRESHOLD, RERANK_METHOD
        from src.task10_generation import LLM_MODEL, TEMPERATURE, TOP_P

        config_rows = [
            ("Chunking", CHUNKING_METHOD, "Cách cắt văn bản thành đoạn nhỏ"),
            ("Chunk size / overlap", f"{CHUNK_SIZE} / {CHUNK_OVERLAP}", "Độ dài mỗi đoạn & phần lặp giữa 2 đoạn"),
            ("Embedding model", EMBEDDING_MODEL, "Model chuyển văn bản → vector"),
            ("Embedding dim", str(EMBEDDING_DIM), "Số chiều vector"),
            ("Vector store", VECTOR_STORE, "Nơi lưu trữ vector"),
            ("Rerank method", RERANK_METHOD.upper(), "Cách gộp/xếp hạng kết quả"),
            ("Fallback threshold", str(SCORE_THRESHOLD), "Điểm cosine tối thiểu trước khi fallback PageIndex"),
            ("LLM model", LLM_MODEL, "Model sinh câu trả lời"),
            ("Temperature", str(TEMPERATURE), "Độ sáng tạo của câu trả lời (thấp = bám sát nguồn)"),
            ("Top-p", str(TOP_P), "Nucleus sampling"),
        ]

        with st.expander("📋 Xem chi tiết cấu hình", expanded=False):
            for label, value, desc in config_rows:
                st.markdown(
                    f'<div class="cfg-row"><span class="cfg-label">{label}</span>'
                    f'<span class="cfg-value">{value}</span></div>'
                    f'<div class="cfg-desc">{desc}</div>',
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.caption(f"⚠ Không đọc được config: {e}")

    st.divider()
    st.caption("**Kiến trúc hệ thống:**")
    st.caption("Hybrid Retrieval (Semantic + BM25) → RRF Rerank → PageIndex Fallback → LLM Generation có Citation")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Load custom CSS
from src.ui_components import load_css, render_chat_message, render_follow_up_chips, render_feedback_buttons, render_trace_toggle, render_trace_drawer
load_css()

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.markdown(
    '''
    <div class="shopee-hero">
      <div class="shopee-hero-icon">🛍️</div>
      <div>
        <p class="shopee-hero-title">Shopee Trợ Giúp — Trợ lý AI</p>
        <p class="shopee-hero-sub">Hỏi đáp chính sách thương mại điện tử &amp; hỗ trợ khách hàng, có trích dẫn nguồn</p>
      </div>
    </div>
    ''',
    unsafe_allow_html=True,
)

# Render active trace drawer if any
# We do this at the top level so it acts as an overlay
for msg in st.session_state.messages:
    if msg.get("role") == "assistant" and st.session_state.get("show_trace") == msg.get("id"):
        # Mock trace steps for the demo
        steps = [
            {"title": "Phân tích ý định", "duration": "180ms", "tags": ["intent: " + msg.get("query", "")[:10]], "details": "Phân tích từ khóa và ngữ cảnh câu hỏi."},
            {"title": "Truy xuất dữ liệu (Hybrid)", "duration": "1.2s", "tags": ["hybrid", f"top_k={top_k}"], "details": "Kết hợp Dense Search (bge-m3) và Sparse Search (BM25) với RRF."},
            {"title": "Tổng hợp câu trả lời", "duration": "3.5s", "tags": ["llm", "generation"], "details": "Sinh câu trả lời qua OpenRouter API kèm trích dẫn (citation)."}
        ]
        render_trace_drawer(msg["id"], steps)

# Hiển thị lịch sử chat
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        render_chat_message("user", msg["content"], msg_id=msg.get("id"))
    else:
        render_chat_message("assistant", msg["content"], msg_id=msg.get("id"), sources=msg.get("sources"), time_taken=msg.get("time_taken", 0))
        # Render interactive buttons under the message
        st.write("") # small spacing
        render_trace_toggle(msg["id"])
        if i == len(st.session_state.messages) - 1: # Only show followups for latest message
            render_follow_up_chips(["Thanh toán thế nào?", "Hướng dẫn đổi trả", "Làm sao mua hàng nước ngoài?"])
        render_feedback_buttons(msg["id"])
        st.write("---")

# =============================================================================
# QUERY HANDLING
# =============================================================================

# For styling the composer to look like the design, we'd need more CSS hacks for st.chat_input,
# but we stick to st.chat_input for functionality.
user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/hỗ trợ e-commerce...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    msg_id = str(uuid.uuid4())

    # Hiển thị câu hỏi của user
    user_msg = {"role": "user", "content": query, "id": msg_id + "_u"}
    st.session_state.messages.append(user_msg)
    
    # We must rerun to render the user message properly before starting the generation
    st.rerun()

elif len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
    # If the last message was from user, generate response
    query = st.session_state.messages[-1]["content"]
    msg_id = st.session_state.messages[-1]["id"].replace("_u", "_b")
    
    with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
        start_time = time.time()
        try:
            from src.task10_generation import generate_with_citation
            response = generate_with_citation(query, top_k=top_k)
            answer = response.get("answer", "Chưa thể trả lời.")
            sources = response.get("sources", [])
        except NotImplementedError:
            answer = "⚠️ **Task 10 chưa được implement.**"
            sources = []
        except Exception as e:
            answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
            sources = []
        time_taken = time.time() - start_time

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "time_taken": time_taken,
        "id": msg_id,
        "query": query
    })
    st.rerun()

