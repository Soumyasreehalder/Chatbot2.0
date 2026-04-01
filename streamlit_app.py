from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import requests
import streamlit as st

# --------------------------------------
# Config
# --------------------------------------
st.set_page_config(page_title="OPS Pilot AI Agent", page_icon="📘", layout="centered")

FASTAPI_CHAT_URL = os.getenv("FASTAPI_CHAT_URL", "http://localhost:8000/chat")
FASTAPI_RESET_URL = os.getenv("FASTAPI_RESET_URL", "http://localhost:8000/reset")

# --------------------------------------
# Session state
# --------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sop_session_id" not in st.session_state:
    st.session_state.sop_session_id = None
if "initialized" not in st.session_state:
    st.session_state.initialized = True

# --------------------------------------
# Inject CSS for ChatGPT-like layout
# --------------------------------------
st.markdown("""
<style>
/* Import a clean font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Full-height layout */
.stApp {
    background-color: #ffffff;
    color: #111111;
}

/* Fixed compact header */
.omni-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 999;
    background: #ffffff;
    border-bottom: 1px solid #e5e5e5;
    padding: 10px 0 8px 0;
    text-align: center;
}
.omni-header.large {
    padding: 40px 0 16px 0;
    border-bottom: none;
    background: transparent;
}
.omni-header h1 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111111;
    margin: 0;
    letter-spacing: 0.02em;
}
.omni-header.large h1 {
    font-size: 2.2rem;
    color: #111111;
}
.omni-header p {
    font-size: 0.72rem;
    color: #666;
    margin: 2px 0 0 0;
}
.omni-header.large p {
    font-size: 0.95rem;
}

/* Chat container with padding to avoid header/input overlap */
.chat-container {
    margin-top: 70px;
    margin-bottom: 90px;
    padding: 0 8px;
    max-width: 720px;
    margin-left: auto;
    margin-right: auto;
}

/* No-chat hero */
.hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
}
.hero h2 {
    font-size: 2rem;
    font-weight: 600;
    color: #ececec;
    margin-bottom: 8px;
}
.hero p {
    color: #888;
    font-size: 1rem;
}

/* Anchor for auto-scroll */
#chat-bottom { height: 1px; }

/* Make chat input bar sticky at bottom */
.stChatInputContainer, [data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 20px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(720px, 92vw) !important;
    background: #f4f4f4 !important;
    border-radius: 16px !important;
    border: 1px solid #ddd !important;
    z-index: 998 !important;
    padding: 4px 8px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important;
}

/* Override Streamlit default chat input colors */
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #111111 !important;
    font-size: 0.97rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #999 !important;
}

/* Utility buttons strip */
.btn-strip {
    position: fixed;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 10px;
    z-index: 997;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------
# Dynamic Header — compact if chat started
# --------------------------------------
has_messages = len(st.session_state.messages) > 0
header_class = "omni-header" if has_messages else "omni-header large"

st.markdown(f"""
<div class="{header_class}">
  <h1>🤖 Ops Pilot AI Agent </h1>
  <p>Think deeper. Resolve faster. Operate smarter.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------
# Chat area
# --------------------------------------
if not has_messages:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;min-height:70vh;text-align:center;flex-direction:column;">
      <p style="color:#666;font-size:1.05rem;margin-top:80px;">Describe your issue below to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        # Add anchor id on every user message so we can scroll to the latest one
        if msg["role"] == "user":
            st.markdown(f'<div id="user-msg-{i}"></div>', unsafe_allow_html=True)
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Find index of the last user message to scroll to
    last_user_idx = max(
        (i for i, m in enumerate(st.session_state.messages) if m["role"] == "user"),
        default=None
    )

    if last_user_idx is not None:
        st.markdown(f"""
        <script>
        (function() {{
            var targetId = 'user-msg-{last_user_idx}';
            var attempts = 0;
            var interval = setInterval(function() {{
                var el = document.getElementById(targetId);
                if (el) {{
                    var rect = el.getBoundingClientRect();
                    var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                    var targetPos = rect.top + scrollTop - 75;
                    window.scrollTo({{ top: targetPos, behavior: 'smooth' }});
                    clearInterval(interval);
                }} else if (attempts > 25) {{
                    clearInterval(interval);
                }}
                attempts++;
            }}, 100);
        }})();
        </script>
        """, unsafe_allow_html=True)

# --------------------------------------
# Chat input
# --------------------------------------
user_prompt = st.chat_input("Describe your issue...")
if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    text = (user_prompt or "").strip()

    quick_replies = {
        frozenset({"hi", "hello", "hey"}): "Hi! 👋 How can I help you today?",
        frozenset({"ok", "okay", "k", "thanks", "thank you", "thx"}): "Happy to help!",
        frozenset({"yeah"}): "Thanks for your response, happy to help you.",
        frozenset({"no", "nah"}): (
            "Sorry to hear that! If the issue continues, you can raise a ServiceNow ticket anytime — I'm here to help! "
            "👉 https://www.servicenow.com/"
        ),
    }

    matched = None
    for keys, reply in quick_replies.items():
        if text.lower() in keys:
            matched = reply
            break

    if matched:
        st.session_state.messages.append({"role": "assistant", "content": matched})
        st.rerun()
    elif not text:
        st.session_state.messages.append({"role": "assistant", "content": "Please describe the issue."})
        st.rerun()
    else:
        payload = {"session_id": st.session_state.get("sop_session_id"), "message": text}

        with st.spinner("Searching SOP and drafting an answer..."):
            try:
                resp = requests.post(FASTAPI_CHAT_URL, json=payload, timeout=90)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "session_id" in data:
                        st.session_state.sop_session_id = data["session_id"]

                    answer = ""
                    if isinstance(data, dict):
                        answer = data.get("answer") or ""
                        if not answer and "detail" in data:
                            answer = f"Backend error: {data['detail']}"
                    if not answer:
                        answer = "No answer returned from SOP backend."

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    try:
                        data = resp.json()
                        err_msg = data.get("detail") or data.get("error") or f"HTTP {resp.status_code}"
                    except Exception:
                        err_msg = f"HTTP {resp.status_code}: {resp.text}"
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {err_msg}"})
            except requests.exceptions.RequestException as e:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Couldn't reach backend. Details: {e}"}
                )

        st.rerun()

# --------------------------------------
# Utility buttons — compact strip
# --------------------------------------
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.sop_session_id = None
        st.rerun()
with col2:
    if st.session_state.sop_session_id:
        if st.button("♻️ Reset SOP session"):
            try:
                url = f"{FASTAPI_RESET_URL}/{st.session_state.sop_session_id}"
                r = requests.post(url, timeout=15)
                if r.status_code == 200:
                    st.session_state.sop_session_id = None
                    st.success("SOP session reset.")
                else:
                    st.warning(f"Reset failed: {r.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Reset request failed: {e}")