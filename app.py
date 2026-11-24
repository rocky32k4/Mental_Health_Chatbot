"""
app.py
------
Streamlit Frontend for the Mental Health Chatbot

Run:
    streamlit run app.py
"""

import sys
import os
import streamlit as st

# Add src/ to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.rag_chain   import chat
from src.vectorstore import get_status


# ─────────────────────────────────────────────
# PAGE CONFIG
# Must be the first Streamlit command
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "Mental Health Support",
    page_icon  = "🧠",
    layout     = "centered",
)


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Crisis banner */
    .crisis-banner {
        background-color: #fff3cd;
        border-left: 5px solid #ff6b6b;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: #f0f2f6;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #555;
        margin-bottom: 16px;
    }

    /* Source tag */
    .source-tag {
        background-color: #e8f4f8;
        color: #2c7be5;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# Streamlit reruns on every interaction
# Session state persists across reruns
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []       # full chat history

if "vectorstore_ready" not in st.session_state:
    status = get_status()
    st.session_state.vectorstore_ready = status["status"] == "ready"
    st.session_state.vectorstore_count = status.get("count", 0)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🧠 Mental Health Support")
st.caption("A compassionate AI assistant grounded in psychology literature.")


# ─────────────────────────────────────────────
# CRISIS BANNER
# Always visible at the top
# ─────────────────────────────────────────────

st.markdown("""
<div class="crisis-banner">
    🆘 <strong>In crisis?</strong> &nbsp;
    iCall: <strong>9152987821</strong> &nbsp;|&nbsp;
    Vandrevala Foundation: <strong>1860-2662-345</strong> (24/7) &nbsp;|&nbsp;
    Emergency: <strong>112</strong>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────

st.markdown("""
<div class="disclaimer">
    ⚠️ This chatbot provides <strong>psychoeducation and emotional support only</strong>.
    It is <strong>not a substitute</strong> for professional mental health care.
    If you are struggling, please consult a licensed therapist or doctor.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VECTOR STORE STATUS CHECK
# ─────────────────────────────────────────────

if not st.session_state.vectorstore_ready:
    st.error(
        "⚠️ Knowledge base not found.\n\n"
        "Please run the following commands first:\n"
        "```\n"
        "python src/ingestion.py\n"
        "python src/vectorstore.py\n"
        "```"
    )
    st.stop()  # stop rendering rest of the page


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This assistant uses **Retrieval-Augmented Generation (RAG)**
    to answer questions based on verified psychology literature.

    **Knowledge Base:**
    - Motivational Interviewing for Clinical Practice
    - Suicide Prevention Handbook (SPIF)

    **Model:** LLaMA 3.3 70B via Groq
    """)

    st.divider()

    # Status indicator
    st.markdown(f"**Knowledge base:** {st.session_state.vectorstore_count} chunks loaded ✅")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("""
    **Crisis Resources (India)**
    - iCall: 9152987821
    - Vandrevala: 1860-2662-345
    - SNEHI: 044-24640050
    - Emergency: 112
    """)


# ─────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────

# Show welcome message if no history yet
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello! I'm here to offer support and information about mental health. "
            "How are you feeling today? Feel free to share what's on your mind."
        )

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources under assistant messages if available
        if message["role"] == "assistant" and message.get("sources"):
            source_html = " ".join(
                f'<span class="source-tag">📚 {s}</span>'
                for s in message["sources"]
            )
            st.markdown(source_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────

user_input = st.chat_input("Type your message here...")

if user_input:

    # ── Show user message ──
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add to history
    st.session_state.messages.append({
        "role"   : "user",
        "content": user_input,
    })

    # ── Get bot response ──
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Build chat history in the format rag_chain expects
            # Only pass role + content (not sources)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # exclude current message
            ]

            result = chat(user_input, history)

        # Show response
        st.markdown(result["response"])

        # Show sources if available
        if result["sources"]:
            source_html = " ".join(
                f'<span class="source-tag">📚 {s}</span>'
                for s in result["sources"]
            )
            st.markdown(source_html, unsafe_allow_html=True)

        # Show crisis badge if triggered
        if result["is_crisis"]:
            st.error("🆘 Crisis resources have been shared above. Please reach out.")

    # Add bot response to history
    st.session_state.messages.append({
        "role"   : "assistant",
        "content": result["response"],
        "sources": result["sources"],
    })