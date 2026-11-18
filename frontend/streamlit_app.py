# frontend/streamlit_app.py
import sys
import os
import tempfile
import time
import re
import json
import logging
from typing import Dict, Any, List

import streamlit as st

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from frontend.components.session_manager import SessionManager
from frontend.components.graph_view import render_knowledge_graph
from app.database.vector_store import UnifiedVectorStore
from app.services.llm_service import get_llm, generate_with_retry
from app.services.style_detector import detect_style_from_text
from app.agents.ingestion_agent import IngestionAgent
from app.agents.quiz_agent import QuizAgent
from app.agents.revision_agent import RevisionAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.organizer_agent import OrganizerAgent
from app.utils.scheduler import get_due_topics
from app.config import PROJECT_NAME, VERSION, GEMINI_MODEL

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{PROJECT_NAME} — Agentic Knowledge Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Modern Custom CSS Styling
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    /* Gradient Background Accents */
    .stApp {
        background: radial-gradient(circle at top right, #131b2e 0%, #0b0f19 55%, #07090e 100%);
    }

    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF 0%, #C7D2FE 40%, #818CF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        font-weight: 400;
        margin: 0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(19, 27, 46, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(8px);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .metric-lbl {
        color: #94A3B8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .pill-purple {
        background: rgba(139, 92, 246, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    .pill-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        transform: translateY(-1px);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 10px 18px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }

    /* Hide standard Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Initialize Singletons & Session
# ------------------------------------------------------------------------------
@st.cache_resource
def init_core_services():
    vector_store = UnifiedVectorStore()
    ingestion = IngestionAgent()
    quiz = QuizAgent()
    revision = RevisionAgent()
    planner = PlannerAgent()
    organizer = OrganizerAgent()
    return vector_store, ingestion, quiz, revision, planner, organizer

vector_store, ingestion_agent, quiz_agent, revision_agent, planner_agent, organizer_agent = init_core_services()
session = SessionManager()

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "_last_quiz" not in st.session_state:
    st.session_state._last_quiz = ""
if "_last_evaluation" not in st.session_state:
    st.session_state._last_evaluation = ""
if "_last_revision" not in st.session_state:
    st.session_state._last_revision = ""
if "last_retrieval_context" not in st.session_state:
    st.session_state.last_retrieval_context = ""
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

# ------------------------------------------------------------------------------
# Sidebar: System Monitor & Quick Info
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 MemoryPalAI")
    st.caption(f"v{VERSION} • Agentic Learning Workspace")

    st.markdown("---")
    st.markdown("#### ⚡ System Telemetry")

    # Backend indicator
    backend_name = vector_store.backend.title()
    if vector_store.backend == "pinecone":
        st.markdown(f'<span class="status-pill pill-purple">☁️ Pinecone (Cloud)</span>', unsafe_allow_html=True)
    elif vector_store.backend == "chroma":
        st.markdown(f'<span class="status-pill pill-green">💾 ChromaDB (Persistent)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-pill pill-amber">⚠️ Fallback Store</span>', unsafe_allow_html=True)

    # LLM Model indicator
    st.markdown(f'<span class="status-pill pill-blue">🤖 {GEMINI_MODEL}</span>', unsafe_allow_html=True)

    stats = vector_store.list_documents()
    total_chunks = stats.get("total_chunks", 0)
    st.metric(label="Total Indexed Vectors", value=total_chunks)

    st.markdown("---")
    st.markdown("#### 📁 Active Uploads")
    user_files = session.list_files()
    if user_files:
        for f in user_files:
            st.markdown(f"- `{os.path.basename(f)}`")
        if st.button("🧹 Clear Session Files", key="clear_session_btn"):
            session.clear()
            st.rerun()
    else:
        st.caption("No files uploaded in this session yet.")

    st.markdown("---")
    st.markdown("#### ⚙️ Quick Settings")
    st.caption("Environment configured via `.env` / Server variables.")
    if st.button("🔄 Reload Vector Store"):
        st.cache_resource.clear()
        st.rerun()

# ------------------------------------------------------------------------------
# Top Hero Header
# ------------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🧠 MemoryPalAI Workspace</div>
    <p class="hero-subtitle">
        Intelligent Knowledge Extractor • Style-Aware Grounded RAG • Adaptive Quiz Arena • Spaced Repetition Companion
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Navigation Tabs
# ------------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "📥 Ingestion Hub",
    "💡 AI Tutor & Roadmap",
    "🎯 Quiz & Revision Studio",
    "🕸️ Knowledge Graph & Memory"
])

# ==============================================================================
# TAB 1: 📊 Overview & Dashboard
# ==============================================================================
with tabs[0]:
    # 4 Quick Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        doc_count = stats.get("total_documents", len(user_files))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{doc_count}</div>
            <div class="metric-lbl">Documents Ingested</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_chunks}</div>
            <div class="metric-lbl">Vector Embeddings</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        mem_profile = revision_agent._load_memory()
        total_quizzes = sum(v.get("attempts", 0) for v in mem_profile.values() if isinstance(v, dict))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_quizzes}</div>
            <div class="metric-lbl">Quizzes Attempted</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        all_avgs = [v.get("avg_score", 0) for v in mem_profile.values() if isinstance(v, dict) and isinstance(v.get("avg_score"), (int, float))]
        overall_retention = round(sum(all_avgs) / len(all_avgs), 1) if all_avgs else 100.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{overall_retention}%</div>
            <div class="metric-lbl">Knowledge Retention</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Spaced Repetition Due Today Alert Card
    due_topics = get_due_topics(mem_profile)
    if due_topics:
        st.markdown("""
        <div class="glass-card" style="border-left: 4px solid #F59E0B;">
            <h4 style="color: #FBBF24; margin-top:0;">⚡ Spaced Repetition Due For Review Today</h4>
            <p style="color: #CBD5E1; font-size: 0.92rem;">
                Based on your past quiz performances, these topics require reinforcement to lock into long-term memory:
            </p>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(min(len(due_topics), 3))
        for idx, item in enumerate(due_topics[:3]):
            with cols[idx]:
                st.info(f"**{item['topic']}**\n- Avg Score: {item['avg_score']}%\n- Status: {item['status']}")
    else:
        st.success("🎉 All caught up! No critical topics pending spaced repetition review today.")

    st.markdown("---")
    st.markdown("### 🚀 Fast Launchpad")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("#### 1. Ingest Documents")
        st.caption("Upload PDFs, audio lectures, TXT, DOCX, or paste URLs to extract style, tone, and vector embeddings.")
    with colB:
        st.markdown("#### 2. Ask Grounded Questions")
        st.caption("Receive answers strictly cited from your study notes, accompanied by an AI 3-phase study roadmap.")
    with colC:
        st.markdown("#### 3. Test & Auto-Revise")
        st.caption("Generate active MCQs. If you score under 70%, MemoryPalAI triggers personalized revision flashcards.")


# ==============================================================================
# TAB 2: 📥 Knowledge Ingestion Hub
# ==============================================================================
with tabs[1]:
    st.markdown("### 📥 Ingest Knowledge & Manage Workspace")
    st.caption("Supported formats: **PDF, TXT, DOCX, MP3, WAV, M4A**, or direct web URLs.")

    col_up1, col_up2 = st.columns([1.6, 1.2])

    with col_up1:
        st.markdown("#### 📤 Upload Material")
        uploaded_file = st.file_uploader(
            "Select study file to process and embed",
            type=["txt", "pdf", "docx", "mp3", "wav", "m4a"],
            accept_multiple_files=False
        )
        url_input = st.text_input(
            "Or enter a public URL / Article link:",
            placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        topic_custom = st.text_input(
            "Optional topic label (overrides auto-detection):",
            placeholder="e.g. Distributed Systems"
        )

        process_button = st.button("🚀 Process & Ingest Material", key="btn_ingest_now")

    with col_up2:
        st.markdown("#### 🔍 Extracted Metadata Preview")
        preview_box = st.empty()
        preview_box.info("Upload a document or provide a URL, then click 'Process & Ingest Material' to view style tags and chunk statistics.")

    if process_button:
        if not uploaded_file and not url_input:
            st.error("Please select a file or provide a web URL.")
        else:
            with st.spinner("Processing document, extracting text & style..."):
                raw_text = ""
                source_name = ""
                temp_dir = tempfile.mkdtemp()

                if uploaded_file:
                    source_name = uploaded_file.name
                    file_path = os.path.join(temp_dir, source_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # If audio, provide audio player preview
                    if source_name.lower().endswith((".mp3", ".wav", ".m4a")):
                        st.audio(uploaded_file)

                    docs = ingestion_agent.ingest(file_path)
                    if docs:
                        raw_text = "\n\n".join([d.page_content for d in docs])
                elif url_input:
                    source_name = url_input
                    import requests
                    try:
                        resp = requests.get(url_input, timeout=12)
                        raw_text = resp.text
                    except Exception as e:
                        st.error(f"Failed to fetch URL: {e}")

                if not raw_text or not raw_text.strip():
                    st.error("No extractable text found in this source.")
                else:
                    # Detect style and topic
                    style_info = detect_style_from_text(raw_text[:3000])
                    subject = topic_custom.strip() if topic_custom and topic_custom.strip() else style_info.get("subject", "General")
                    style = style_info.get("style", "Descriptive")
                    tone = style_info.get("tone", "Neutral")
                    tags = style_info.get("tags", [])

                    with preview_box.container():
                        st.markdown(f"**Source:** `{source_name}`")
                        st.markdown(f"**Detected Subject:** `{subject}`")
                        st.markdown(f"""
                        <span class="status-pill pill-purple">Style: {style}</span>
                        <span class="status-pill pill-blue">Tone: {tone}</span>
                        """, unsafe_allow_html=True)
                        if tags:
                            tag_html = " ".join([f'<span class="status-pill pill-green">#{t}</span>' for t in tags[:5]])
                            st.markdown(f"<div style='margin-top:8px;'>{tag_html}</div>", unsafe_allow_html=True)

                    # Ingest into Unified Vector Store
                    metadata = {
                        "source": source_name,
                        "subject": subject,
                        "style": style,
                        "tone": tone,
                        "tags": tags,
                    }

                    res = vector_store.add_document(
                        doc_id=os.path.basename(source_name),
                        content=raw_text,
                        metadata=metadata,
                        topic=subject
                    )

                    if res and res.get("status") == "duplicate":
                        st.warning("⚠️ This document content is already indexed. Duplicate upload skipped.")
                    elif res and res.get("status") == "success":
                        st.success(f"✅ Successfully indexed `{source_name}` into {res.get('backend').title()} ({res.get('chunks')} chunks).")
                        session.add_file(source_name)
                    else:
                        st.info(f"Ingested locally for session: `{source_name}`")
                        session.add_file(source_name)

    st.markdown("---")
    st.markdown("#### 📑 Indexed Documents in Active Vector Store")
    current_docs = stats.get("documents", [])
    if current_docs:
        for d in current_docs:
            col_d1, col_d2, col_d3 = st.columns([3, 1.5, 1])
            with col_d1:
                st.markdown(f"📄 **{d.get('document_id')}** — `{d.get('topic')}`")
            with col_d2:
                st.caption(f"{d.get('chunks')} chunks • {d.get('indexed_at')}")
            with col_d3:
                if st.button("Delete", key=f"del_{d.get('document_id')}"):
                    vector_store.delete_document(d.get('document_id'))
                    session.remove_file(d.get('document_id'))
                    st.success(f"Deleted {d.get('document_id')}")
                    st.rerun()
    else:
        st.info("No documents currently listed in vector store index.")


# ==============================================================================
# TAB 3: 💡 AI Learning Companion & Grounded RAG
# ==============================================================================
with tabs[2]:
    st.markdown("### 💡 Grounded AI Tutor & 3-Phase Mastery Roadmap")
    st.caption("Ask questions about your uploaded materials. The assistant answers grounded strictly on retrieved context.")

    col_q1, col_q2 = st.columns([3, 1.2])
    with col_q1:
        user_query = st.text_input("What concept would you like to explore or learn?", placeholder="e.g. How does backpropagation compute gradients?")
    with col_q2:
        available_topics = stats.get("topics", [])
        topic_options = ["All Topics"] + available_topics
        chosen_topic = st.selectbox("Filter by topic (optional):", options=topic_options)

    if st.button("🔎 Ask Grounded AI Tutor", key="btn_ask_rag"):
        if not user_query or not user_query.strip():
            st.error("Please enter a question.")
        else:
            filter_arg = None if chosen_topic == "All Topics" else chosen_topic
            with st.spinner("Searching knowledge base & synthesizing grounded response..."):
                retrieval = vector_store.query(user_query, top_k=4, topic_filter=filter_arg)
                docs = retrieval.get("documents", [])
                metas = retrieval.get("metadatas", [])
                llm = get_llm()

                st.session_state.last_query = user_query

                if not docs:
                    # Fallback to general AI knowledge
                    st.warning("⚠️ No matching context found in your indexed documents. Supplying general knowledge response...")
                    gen_prompt = f"You are MemoryPalAI tutor. Answer this question clearly and educationally: {user_query}"
                    answer = generate_with_retry(llm, gen_prompt)
                    st.session_state.last_answer = answer
                    st.session_state.last_retrieval_context = ""
                else:
                    context = "\n\n---\n\n".join(docs[:4])
                    st.session_state.last_retrieval_context = context

                    grounded_prompt = f"""
You are MemoryPalAI — an expert retrieval-grounded AI learning companion.
Answer the following QUESTION based on the provided CONTEXT.

CONTEXT:
{context}

QUESTION:
{user_query}

Provide a clear, pedagogical, structured explanation with key points and definitions.
"""
                    answer = generate_with_retry(llm, grounded_prompt)
                    st.session_state.last_answer = answer

                # Generate 3-Phase Roadmap
                context_summary = st.session_state.last_retrieval_context[:1000] if st.session_state.last_retrieval_context else user_query
                roadmap = planner_agent.create_plan(
                    user_goal=f"Master {user_query}",
                    user_knowledge_summary=context_summary,
                    subject=chosen_topic if chosen_topic != "All Topics" else (metas[0].get("subject", "General") if metas else "General"),
                    style=metas[0].get("style", "Descriptive") if metas else "Descriptive"
                )
                st.session_state._last_plan = roadmap
                st.session_state._last_subject = chosen_topic if chosen_topic != "All Topics" else (metas[0].get("subject", "General") if metas else "General")

                # Store history
                st.session_state.history.append({
                    "query": user_query,
                    "response": answer,
                    "time": time.time()
                })

    # Render Answer & Roadmap if available
    if st.session_state.last_answer:
        st.markdown("---")
        st.markdown(f"#### 🧠 AI Tutor Answer for: *\"{st.session_state.last_query}\"*")
        st.markdown(st.session_state.last_answer)

        # Retrieved Context Accordion
        if st.session_state.last_retrieval_context:
            with st.expander("🔍 Inspect Grounded Evidence & Retrieved Chunks"):
                st.markdown(st.session_state.last_retrieval_context)

        # Roadmap Card
        if st.session_state.get("_last_plan"):
            st.markdown("---")
            st.markdown("#### 🗺️ Suggested 3-Phase Study Roadmap & Video Resources")
            st.markdown(st.session_state._last_plan)


# ==============================================================================
# TAB 4: 🎯 Adaptive Quiz & Revision Studio
# ==============================================================================
with tabs[3]:
    st.markdown("### 🎯 Adaptive Quiz & Revision Studio")
    st.caption("Test your recall on any indexed topic or recent study context. If you score under 70%, smart revision notes are generated automatically.")

    col_qz1, col_qz2 = st.columns([2.5, 1])
    with col_qz1:
        quiz_subject = st.text_input(
            "Topic or Subject for Quiz:",
            value=st.session_state.get("_last_subject", "General Knowledge")
        )
    with col_qz2:
        num_q = st.select_slider("Number of Questions:", options=[3, 4, 5, 8], value=4)

    if st.button("⚡ Generate Interactive Quiz", key="btn_gen_quiz_arena"):
        with st.spinner("Designing conceptual MCQs..."):
            context_for_quiz = st.session_state.get("last_retrieval_context", "")
            raw_quiz = quiz_agent.generate_quiz(
                subject=quiz_subject,
                roadmap=context_for_quiz,
                num_questions=num_q,
                user_goal="Assess mastery"
            )
            st.session_state._last_quiz = raw_quiz
            st.session_state._last_subject = quiz_subject
            st.session_state._last_evaluation = ""
            st.session_state._last_revision = ""

    # Render Quiz if generated
    if st.session_state.get("_last_quiz"):
        st.markdown("---")
        st.markdown(f"### 📝 Multiple Choice Quiz: **{st.session_state.get('_last_subject')}**")

        quiz_blocks = [b.strip() for b in re.split(r"-{3,}", st.session_state._last_quiz) if b.strip()]
        answers_user = {}

        with st.form("interactive_quiz_form"):
            for i, block in enumerate(quiz_blocks):
                lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                if not lines:
                    continue
                q_title = lines[0]
                options = []
                for ln in lines[1:]:
                    m = re.match(r"^[A-D]\s*[\)\.]\s*(.+)", ln)
                    if m:
                        options.append(m.group(1).strip())
                    elif ln:
                        options.append(ln)

                while len(options) < 4:
                    options.append(f"Option {len(options)+1}")

                st.markdown(f"**{q_title}**")
                choice = st.radio(
                    f"Select answer for Q{i+1}",
                    options=options,
                    key=f"quiz_opt_{i}",
                    label_visibility="collapsed"
                )
                answers_user[f"Q{i+1}"] = choice

            submit_quiz = st.form_submit_button("📊 Submit Answers for Evaluation")

        if submit_quiz:
            # Map selected option back to A, B, C, D
            letter_mapping = {}
            for i, block in enumerate(quiz_blocks):
                lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
                options = []
                for ln in lines[1:]:
                    m = re.match(r"^[A-D]\s*[\)\.]\s*(.+)", ln)
                    if m:
                        options.append(m.group(1).strip())
                    elif ln:
                        options.append(ln)
                while len(options) < 4:
                    options.append(f"Option {len(options)+1}")

                user_choice_text = answers_user.get(f"Q{i+1}", options[0])
                if user_choice_text in options:
                    idx = options.index(user_choice_text)
                    letter_mapping[f"Q{i+1}"] = ["A", "B", "C", "D"][idx]
                else:
                    letter_mapping[f"Q{i+1}"] = "A"

            with st.spinner("Evaluating your quiz answers..."):
                eval_result = quiz_agent.evaluate_answers(
                    subject=st.session_state.get("_last_subject", "General"),
                    user_answers=letter_mapping,
                    quiz_text=st.session_state._last_quiz
                )
                st.session_state._last_evaluation = eval_result

                # Parse percentage score
                pct = None
                try:
                    m = re.search(r"(\d+)\s*(?:out of|/)\s*(\d+)", eval_result)
                    if m:
                        score_val = int(m.group(1))
                        total_val = int(m.group(2))
                        pct = (score_val / total_val) * 100
                except Exception:
                    pct = None

                # Record score to user profile memory
                if pct is not None:
                    revision_agent.record_quiz_result(
                        subject=st.session_state.get("_last_subject", "General"),
                        score_pct=pct,
                        evaluation_text=eval_result
                    )

                # Auto-Revision Trigger
                if pct is not None and pct < 70.0:
                    rev_notes = revision_agent.revise(
                        subject=st.session_state.get("_last_subject", "General"),
                        evaluation_text=eval_result
                    )
                    st.session_state._last_revision = rev_notes

    # Render Evaluation Scorecard & Revision Pack
    if st.session_state.get("_last_evaluation"):
        st.markdown("---")
        st.markdown(st.session_state._last_evaluation)

        if st.session_state.get("_last_revision"):
            st.markdown("---")
            st.markdown("""
            <div class="glass-card" style="border-left: 4px solid #EC4899;">
                <h4 style="color: #F472B6; margin-top:0;">🔁 Auto-Revision Deck Activated (< 70% Score)</h4>
                <p style="color: #CBD5E1; font-size: 0.92rem;">
                    MemoryPalAI has synthesized simplified explanations, video resources, and recall flashcards to reinforce your weak concepts.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state._last_revision)


# ==============================================================================
# TAB 5: 🕸️ Interactive Knowledge Graph & Memory Analytics
# ==============================================================================
with tabs[4]:
    st.markdown("### 🕸️ Interactive Concept Graph & Spaced Repetition Memory")
    st.caption("Visual representation of entities and relationships extracted from your documents.")

    col_g1, col_g2 = st.columns([2.5, 1])
    with col_g1:
        # Build graph from session or sample
        graph_text = st.session_state.get("last_retrieval_context", "")
        if not graph_text and current_docs:
            graph_text = f"Knowledge entities for {current_docs[0].get('document_id')}"

        if graph_text:
            try:
                g_data = organizer_agent.extract_graph_data(graph_text[:2500])
                render_knowledge_graph(g_data, height="520px")
            except Exception as e:
                st.warning(f"Graph visualizer note: {e}")
        else:
            st.info("Ingest documents or ask a question to generate and inspect your knowledge graph.")

    with col_g2:
        st.markdown("#### 📊 Knowledge Retention by Topic")
        mem_data = revision_agent._load_memory()
        if mem_data:
            for top_name, top_val in mem_data.items():
                if isinstance(top_val, dict):
                    avg = top_val.get("avg_score", 0)
                    att = top_val.get("attempts", 0)
                    schedule = top_val.get("review_schedule", {})
                    next_rev = schedule.get("next_review_date", "Pending")

                    st.markdown(f"**{top_name}**")
                    st.progress(min(max(float(avg) / 100.0, 0.0), 1.0))
                    st.caption(f"Avg Score: {avg}% • {att} Quizzes • Next Review: {next_rev}")
                    st.markdown("---")
        else:
            st.caption("No quiz performance history recorded yet.")

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 12px 0;">
    MemoryPalAI v{VERSION} • Agentic AI Personal Learning Companion • Built with Gemini & LangChain
</div>
""", unsafe_allow_html=True)
