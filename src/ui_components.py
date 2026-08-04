import streamlit as st
import json

def load_css():
    """Load the custom CSS overrides"""
    try:
        with open("styles.html", "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def render_chat_message(role: str, content: str, msg_id: str = None, sources: list = None, time_taken: float = 0):
    """Render a static chat message box (User or Bot)"""
    if role == "user":
        st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
    else:
        # Bot message container start
        html = '<div class="msg-bot">'
        
        # In a real scenario we could render pipeline breadcrumbs here
        html += '''
        <div class="pipeline">
          <div class="pipeline-step done"><span class="pip-dot"></span>Hiểu yêu cầu</div>
          <div class="pipeline-line"></div>
          <div class="pipeline-step done"><span class="pip-dot"></span>Truy xuất dữ liệu</div>
          <div class="pipeline-line"></div>
          <div class="pipeline-step done"><span class="pip-dot"></span>Tổng hợp câu trả lời</div>
        </div>
        '''
        
        # Bot text content
        # Note: In Streamlit, rendering markdown inside a div via unsafe_allow_html 
        # means we should convert markdown to HTML first. For simplicity, we just use the raw text if it's plain,
        # or we can rely on st.markdown separately. 
        # But to keep the box shadow around everything, we wrap it in a div.
        html += f'<div class="bot-text">{content}</div>'
        
        # Meta Row
        if sources:
            source_name = sources[0].get("metadata", {}).get("source", "Hybrid Search")
            confidence = "Cao" if len(sources) > 0 else "Thấp"
            html += f'''
            <div class="meta-row">
              <div class="meta-chip chip-confidence">Độ khớp dữ liệu: {confidence} ({len(sources)} chunks)</div>
              <div class="meta-chip chip-source">Nguồn: {source_name}</div>
              <div class="meta-chip chip-time">Phản hồi trong {time_taken:.1f}s</div>
            </div>
            '''
        
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

def render_follow_up_chips(suggestions: list):
    """Render interactive follow-up chips using st.button and CSS marker"""
    cols = st.columns([len(s) for s in suggestions]) # Rough width estimation
    for i, s in enumerate(suggestions):
        with cols[i]:
            # Add marker for CSS targeting
            st.markdown('<span id="marker-followup"></span>', unsafe_allow_html=True)
            if st.button(s, key=f"sug_{s}"):
                st.session_state.pending_query = s
                st.rerun()

def render_feedback_buttons(msg_id: str):
    """Render interactive feedback buttons"""
    # We use a very small column ratio for the buttons
    col1, col2, col3 = st.columns([8, 1, 1])
    with col2:
        # upvote
        marker = "#marker-feedback-up" if st.session_state.get(f"fb_{msg_id}") == "up" else "#marker-feedback"
        st.markdown(f'<span id="{marker[1:]}"></span>', unsafe_allow_html=True)
        if st.button("👍", key=f"up_{msg_id}"):
            st.session_state[f"fb_{msg_id}"] = "up"
            st.rerun()
    with col3:
        # downvote
        marker = "#marker-feedback-down" if st.session_state.get(f"fb_{msg_id}") == "down" else "#marker-feedback"
        st.markdown(f'<span id="{marker[1:]}"></span>', unsafe_allow_html=True)
        if st.button("👎", key=f"down_{msg_id}"):
            st.session_state[f"fb_{msg_id}"] = "down"
            st.rerun()

def render_trace_toggle(msg_id: str):
    """Render the button to open the trace panel"""
    col1, col2 = st.columns([8, 2])
    with col2:
        st.markdown('<span id="marker-trace-toggle"></span>', unsafe_allow_html=True)
        if st.button("🔍 Xem quy trình xử lý", key=f"trace_btn_{msg_id}"):
            st.session_state.show_trace = msg_id
            st.rerun()

def render_trace_drawer(msg_id: str, steps: list):
    """Render the trace drawer overlay if open"""
    if st.session_state.get("show_trace") != msg_id:
        return
        
    # 1. Render the close button in fixed position using a marker
    st.markdown('<span id="marker-drawer-close"></span>', unsafe_allow_html=True)
    if st.button("✖", key="close_drawer_btn"):
        st.session_state.show_trace = None
        st.rerun()

    # 2. Render the actual drawer HTML
    html = '''
    <div class="overlay"></div>
    <div class="drawer open">
      <div class="drawer-head">
        <div class="drawer-title">Quy trình xử lý</div>
      </div>
      <div class="drawer-sub">Chi tiết các bước tạo ra câu trả lời.</div>
    '''
    
    for i, step in enumerate(steps, 1):
        tags_html = "".join([f'<span class="tag">{t}</span>' for t in step.get("tags", [])])
        html += f'''
        <div class="step-card">
          <div class="step-head">
            <div class="step-num">{i}</div>
            <div class="step-title">{step["title"]}</div>
            <div class="step-time">{step.get("duration", "N/A")}</div>
          </div>
          <div class="step-body">
            <div class="step-tags">{tags_html}</div>
            {step.get("details", "")}
          </div>
        </div>
        '''
        
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
