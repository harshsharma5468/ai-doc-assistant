"""
Streamlit UI for the AI Document Assistant.
Features: document upload, multi-turn chat, source viewer, evaluation dashboard.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/ai-doc-assistant",
        "Report a bug": "https://github.com/your-org/ai-doc-assistant/issues",
        "About": "AI Document Assistant — Production RAG Pipeline",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --surface: #1e1e2e;
        --surface-2: #2a2a3e;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --border: rgba(255,255,255,0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .main { background: var(--surface); }

    .stChatMessage {
        background: var(--surface-2) !important;
        border: 1px solid var(--border);
        border-radius: 12px;
        margin-bottom: 8px;
    }

    .source-card {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--primary);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        font-size: 0.85rem;
    }

    .metric-card {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, var(--primary), #8b5cf6);
        transition: width 0.5s ease;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-indexed { background: #065f46; color: #6ee7b7; }
    .badge-pending { background: #78350f; color: #fcd34d; }
    .badge-failed  { background: #7f1d1d; color: #fca5a5; }

    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stButton > button {
        background: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background: var(--primary-dark) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session State ─────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "namespace" not in st.session_state:
    st.session_state.namespace = "default"


# ── API Helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, **kwargs) -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, **kwargs) -> Optional[Dict]:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def check_health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{HEALTH_URL}/live", timeout=5)
        return {"ok": r.status_code == 200}
    except Exception:
        return {"ok": False}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Header
    st.markdown("## 🧠 Doc Assistant")
    st.markdown("---")

    # Health indicator
    health = check_health()
    status_color = "🟢" if health["ok"] else "🔴"
    st.markdown(f"{status_color} API **{'Online' if health['ok'] else 'Offline'}**")
    st.markdown("---")

    # Namespace selector
    st.markdown("### 📁 Workspace")
    namespace = st.text_input("Namespace", value=st.session_state.namespace, key="ns_input")
    if namespace != st.session_state.namespace:
        st.session_state.namespace = namespace
        st.session_state.messages = []
        st.session_state.session_id = None

    # Document upload
    st.markdown("### 📤 Upload Document")
    uploaded = st.file_uploader(
        "PDF, DOCX, TXT, MD, HTML",
        type=["pdf", "docx", "txt", "md", "html"],
        help="Max 50MB",
    )

    if uploaded and st.button("🚀 Index Document", use_container_width=True):
        with st.spinner("Uploading and indexing..."):
            result = None
            try:
                r = requests.post(
                    f"{API_BASE}/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data={"namespace": st.session_state.namespace},
                    timeout=120,
                )
                r.raise_for_status()
                result = r.json()
                st.success(f"✅ Indexed: **{result['filename']}**")
                st.caption(f"ID: `{result['document_id'][:8]}...`")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # Document list
    st.markdown("### 📚 Documents")
    if st.button("🔄 Refresh", key="refresh_sidebar", use_container_width=True):
        data = api_get(f"/documents/?namespace={st.session_state.namespace}")
        if data:
            st.session_state.documents = data.get("documents", [])

    for doc in st.session_state.documents:
        status_badge = {"indexed": "🟢", "pending": "🟡", "failed": "🔴", "processing": "🔵"}.get(
            doc.get("status", ""), "⚪"
        )
        with st.expander(f"{status_badge} {doc.get('filename', 'Unknown')[:25]}"):
            st.caption(f"ID: `{str(doc.get('id', ''))[:8]}`")
            st.caption(f"Chunks: {doc.get('chunk_count', 0)}")
            if st.button("🗑️ Delete", key=f"del_{doc.get('id')}"):
                r = requests.delete(f"{API_BASE}/documents/{doc['id']}", timeout=10)
                if r.status_code == 204:
                    st.success("Deleted")
                    st.rerun()

    st.markdown("---")

    # Search settings
    st.markdown("### ⚙️ Retrieval Settings")
    search_type = st.selectbox("Search Type", ["mmr", "similarity"], index=0)
    top_k = st.slider("Top K chunks", min_value=1, max_value=15, value=5)
    show_sources = st.toggle("Show Sources", value=True)
    show_scores = st.toggle("Show Eval Scores", value=False)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        if st.session_state.session_id:
            requests.delete(f"{API_BASE}/chat/{st.session_state.session_id}", timeout=5)
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()


# ── Main Layout ───────────────────────────────────────────────────────────────

tab_chat, tab_eval, tab_docs = st.tabs(["💬 Chat", "📊 Evaluation", "📋 Documents"])


# ── Chat Tab ──────────────────────────────────────────────────────────────────

with tab_chat:
    st.markdown("### AI Document Assistant")
    st.caption(f"Namespace: `{st.session_state.namespace}` · Session: `{st.session_state.session_id or 'New'}`")

    # Message history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

                if msg.get("sources") and show_sources:
                    with st.expander(f"📎 {len(msg['sources'])} sources"):
                        for src in msg["sources"]:
                            score = src.get("relevance_score", 0)
                            score_pct = int(score * 100)
                            st.markdown(
                                f"""<div class="source-card">
                                    <strong>{src.get('filename', 'Unknown')}</strong>
                                    {'· page ' + str(src['page_number']) if src.get('page_number') else ''}
                                    <div class="score-bar" style="width:{score_pct}%; margin: 6px 0;"></div>
                                    <small style="color:#94a3b8">Relevance: {score:.2%}</small>
                                    <p style="margin-top:8px;font-size:0.8rem;color:#cbd5e1">
                                        {src.get('content_snippet', '')[:200]}...
                                    </p>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                if msg.get("eval_scores") and show_scores:
                    scores = msg["eval_scores"]
                    cols = st.columns(4)
                    for i, (metric, label) in enumerate([
                        ("answer_relevance", "Relevance"),
                        ("faithfulness", "Faithfulness"),
                        ("context_precision", "Precision"),
                        ("context_recall", "Recall"),
                    ]):
                        val = scores.get(metric)
                        with cols[i]:
                            if val is not None:
                                st.metric(label, f"{val:.2%}")
                            else:
                                st.metric(label, "N/A")

                if msg.get("latency_ms"):
                    st.caption(f"⚡ {msg['latency_ms']:.0f}ms")

    # Input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                payload = {
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "namespace": st.session_state.namespace,
                    "search_type": search_type,
                    "top_k": top_k,
                    "include_sources": show_sources,
                    "stream": False,
                }
                result = api_post("/chat/", json=payload)

            if result:
                st.session_state.session_id = result["session_id"]
                answer = result["answer"]
                st.markdown(answer)

                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": answer,
                    "latency_ms": result.get("latency_ms"),
                    "sources": result.get("sources", []),
                    "eval_scores": result.get("evaluation_scores"),
                }
                st.session_state.messages.append(assistant_msg)

                if result.get("sources") and show_sources:
                    with st.expander(f"📎 {len(result['sources'])} sources"):
                        for src in result["sources"]:
                            score = src.get("relevance_score", 0)
                            st.markdown(
                                f"""<div class="source-card">
                                <strong>{src.get('filename')}</strong>
                                {'· p.' + str(src['page_number']) if src.get('page_number') else ''}
                                — Relevance: <strong>{score:.2%}</strong><br/>
                                <small>{src.get('content_snippet', '')[:200]}</small>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                st.caption(
                    f"⚡ {result.get('latency_ms', 0):.0f}ms · "
                    f"Turn {result.get('conversation_turn', 1)} · "
                    f"Model: {result.get('model_used', 'N/A')}"
                )
            else:
                st.error("Failed to get a response. Check that the API is running.")


# ── Evaluation Tab ────────────────────────────────────────────────────────────

with tab_eval:
    st.markdown("### 📊 RAG Evaluation Dashboard")
    st.caption("Run RAGAS evaluation to measure retrieval quality and answer accuracy.")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        eval_questions = st.text_area(
            "Evaluation Questions (one per line)",
            height=180,
            placeholder="What is the main topic of the document?\nHow does X work?\nWhat are the key findings?",
        )
        ground_truths = st.text_area(
            "Ground Truth Answers (optional, one per line)",
            height=120,
            placeholder="Leave empty for reference-free evaluation",
        )
        eval_namespace = st.text_input("Namespace", value=st.session_state.namespace)

    with col_right:
        st.markdown("#### RAGAS Metrics")
        st.markdown("""
        | Metric | Description |
        |--------|-------------|
        | **Answer Relevance** | Is answer on-topic? |
        | **Faithfulness** | Grounded in context? |
        | **Context Precision** | Retrieved chunks relevant? |
        | **Context Recall** | Full coverage of truth? |
        """)

    if st.button("▶️ Run Evaluation", use_container_width=True, type="primary"):
        questions = [q.strip() for q in eval_questions.strip().split("\n") if q.strip()]
        if not questions:
            st.warning("Please enter at least one question.")
        else:
            gts = [g.strip() for g in ground_truths.strip().split("\n") if g.strip()]

            payload: Dict[str, Any] = {
                "questions": questions,
                "namespace": eval_namespace,
                "sample_size": len(questions),
            }
            if gts:
                payload["ground_truths"] = gts

            with st.spinner(f"Evaluating {len(questions)} questions..."):
                result = api_post("/evaluate/", json=payload)

            if result:
                st.success("✅ Evaluation complete!")
                agg = result.get("aggregate_scores", {})
                metrics = [
                    ("Answer Relevance", agg.get("answer_relevance")),
                    ("Faithfulness", agg.get("faithfulness")),
                    ("Context Precision", agg.get("context_precision")),
                    ("Context Recall", agg.get("context_recall")),
                ]
                cols = st.columns(4)
                for i, (label, val) in enumerate(metrics):
                    with cols[i]:
                        if val is not None:
                            delta = f"{val:.2%}"
                            st.metric(label, delta)
                            st.progress(val)
                        else:
                            st.metric(label, "N/A")

                st.markdown("#### Per-Query Results")
                for item in result.get("per_query_scores", []):
                    with st.expander(f"Q: {item['question'][:60]}..."):
                        st.caption(f"**Answer preview:** {item.get('answer_preview', '')[:200]}")
                        scores = item.get("scores", {})
                        sc = st.columns(len(scores))
                        for i, (k, v) in enumerate(scores.items()):
                            with sc[i]:
                                st.metric(k.replace("_", " ").title(), f"{v:.2%}" if v else "N/A")


# ── Documents Tab ─────────────────────────────────────────────────────────────

with tab_docs:
    st.markdown("### 📋 Document Registry")

    col1, col2 = st.columns([3, 1])
    with col1:
        filter_ns = st.text_input("Filter by namespace", value="")
    with col2:
        if st.button("🔄 Refresh", key="refresh_docs_tab", use_container_width=True):
            q = f"/documents/?namespace={filter_ns}" if filter_ns else "/documents/"
            data = api_get(q)
            if data:
                st.session_state.documents = data.get("documents", [])

    if not st.session_state.documents:
        st.info("No documents loaded. Upload a document via the sidebar.")
    else:
        for doc in st.session_state.documents:
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                with c1:
                    st.markdown(f"**{doc.get('filename', 'Unknown')}**")
                    st.caption(f"ID: `{str(doc.get('id', ''))[:12]}`")
                with c2:
                    status = doc.get("status", "unknown")
                    emoji = {"indexed": "✅", "pending": "⏳", "failed": "❌", "processing": "🔄"}.get(status, "❓")
                    st.markdown(f"{emoji} {status}")
                with c3:
                    st.markdown(f"**{doc.get('chunk_count', 0)}** chunks")
                with c4:
                    ns = doc.get("namespace", "default")
                    st.markdown(f"`{ns}`")
                st.markdown("---")
