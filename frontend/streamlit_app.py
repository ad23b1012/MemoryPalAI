# frontend/streamlit_app.py
import sys
import os
import tempfile
import streamlit as st
import hashlib
import json
import time
import re

# make backend importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- FIX 1: REMOVED LANGGRAPH & UNUSED AGENT ---
# from app.services.langgraph_pipeline import build_memorypal_graph, MemoryPalAIState
# from app.agents.retriever_agent import RetrieverAgent 
# --- FIX 2: CORRECTED COMPONENTS PATH ---
from frontend.components.session_manager import SessionManager
from frontend.components.graph_view import render_knowledge_graph

# New imports
from app.services.style_detector import detect_style_from_text
from app.database.pinecone_db import PineconeDB
from app.services.llm_service import get_llm, generate_with_retry
from app.agents.quiz_agent import QuizAgent
from app.agents.revision_agent import RevisionAgent
from app.agents.organizer_agent import OrganizerAgent
from app.services.embedder import embed_text

# Sidebar status
st.sidebar.markdown("### 🧠 MemoryPalAI Status")
st.sidebar.info("Ready for Pinecone-based retrieval and style-aware learning pipeline.")

# Streamlit page config
st.set_page_config(page_title="MemoryPalAI Dashboard 🧠", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 MemoryPalAI — Intelligent Knowledge Workspace")

st.markdown(
    "Upload documents / audio / URLs → extract style/tags → index into Pinecone. "
    "Then go to Retrieve & Learn to ask questions, take quizzes, and get revision loops."
)

# Services & session
session = SessionManager()
organizer = OrganizerAgent()
quiz_agent = QuizAgent()
revision_agent = RevisionAgent()

# Initialize Pinecone safely
pinecone_db = None
try:
    pinecone_db = PineconeDB()
    st.sidebar.success("✅ Pinecone initialized")
except Exception as e:
    pinecone_db = None
    st.sidebar.warning(f"⚠️ Pinecone init failed: {e}")

# session state defaults
if "history" not in st.session_state:
    st.session_state.history = []
if "_last_quiz" not in st.session_state:
    st.session_state._last_quiz = ""
if "_last_evaluation" not in st.session_state:
    st.session_state._last_evaluation = ""
if "last_retrieval_context" not in st.session_state:
    st.session_state.last_retrieval_context = ""
if "generate_quiz_pressed" not in st.session_state:
    st.session_state.generate_quiz_pressed = False

# helper: check whether content already indexed
def content_already_indexed(db, raw_text):
    """
    Use db.has_content if available; otherwise compute a hash and search
    for any recent vectors having the same 'hash' metadata.
    """
    if db is None:
        return False
    if hasattr(db, "has_content") and callable(getattr(db, "has_content")):
        try:
            return db.has_content(raw_text)
        except Exception:
            pass

    # fallback: compute hash and query approximate neighbors to match metadata.hash
    try:
        h = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        snippet = raw_text[:400]
        result = db.query(snippet, top_k=6)
        metas = result.get("metadatas", []) or []
        for m in metas:
            if m.get("hash") == h or m.get("document_hash") == h:
                return True
    except Exception:
        pass
    return False

# two primary tabs
tabs = st.tabs(["📤 Upload & Index", "💬 Retrieve & Learn"])

# -----------------------
# Upload & Index tab
# -----------------------
with tabs[0]:
    st.header("Upload & Index Data")
    st.write("Upload PDFs, text files, or audio. System will extract content, detect style/tone, tag, and index into Pinecone.")
    uploaded_file = st.file_uploader("Choose file (txt, pdf, mp3, wav, m4a)", type=["txt", "pdf", "mp3", "wav", "m4a"], accept_multiple_files=False)

    url_text = st.text_input("Or paste a public URL to fetch text (optional)", placeholder="https://example.com/lecture-notes")
    topic_input = st.text_input("Optional topic tag for this upload:", placeholder="e.g., Artificial Intelligence")

    if st.button("🔁 Process & Index", key="process_index"):
        if not uploaded_file and not url_text:
            st.error("Please upload a file or supply a URL.")
        else:
            temp_dir = tempfile.mkdtemp()
            file_path = None
            raw_text = ""

            if uploaded_file:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.info(f"Saved upload to {file_path}")

            # Extract text
            try:
                if file_path and file_path.lower().endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                elif file_path and file_path.lower().endswith(".pdf"):
                    # use PyMuPDF (fitz) if available
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(file_path)
                        pages = [p.get_text() for p in doc]
                        raw_text = "\n\n".join(pages)
                    except Exception as e:
                        st.error(f"Failed to extract PDF text (need PyMuPDF): {e}")
                        raw_text = ""
                elif url_text:
                    # minimal URL fetch (not robust)
                    import requests
                    r = requests.get(url_text, timeout=10)
                    raw_text = r.text
                else:
                    raw_text = ""
            except Exception as e:
                st.error(f"Failed extracting text: {e}")
                raw_text = ""

            if not raw_text or not raw_text.strip():
                st.error("No text extracted from the provided source.")
            else:
                # detect style/tone/tags (organizer or style detector)
                try:
                    sd = detect_style_from_text(raw_text)
                except Exception:
                    sd = {"subject": topic_input or "Unknown", "style": "Unknown", "tone": "neutral", "tags": []}

                subject = sd.get("subject", topic_input or "Unknown")
                style = sd.get("style")
                tone = sd.get("tone")
                tags = sd.get("tags", [])

                filename = uploaded_file.name if uploaded_file else url_text
                metadata = {
                    "source": filename,
                    "subject": subject,
                    "style": style,
                    "tone": tone,
                    "tags": tags,
                }

                if pinecone_db:
                    already = False
                    try:
                        already = content_already_indexed(pinecone_db, raw_text)
                    except Exception as e:
                        st.warning(f"Dedup check error (continuing): {e}")
                        already = False

                    if already:
                        st.warning("This content appears to already be indexed — skipping duplicate upload.")
                    else:
                        with st.spinner("Indexing into Pinecone..."):
                            resp = pinecone_db.add_document(doc_id=os.path.basename(filename), content=raw_text, metadata=metadata, topic=subject)
                            if resp is None:
                                st.error("Indexing failed. Check server logs.")
                            else:
                                st.success(f"Indexed '{filename}' with subject='{subject}', style='{style}', tone='{tone}'")
                                session.add_file(file_path or filename)
                else:
                    st.info("Pinecone not configured — saving locally to session only.")
                    session.add_file(file_path or filename)

    st.markdown("---")
    st.header("Indexed Documents (summary)")
    if pinecone_db:
        stats = pinecone_db.list_documents()
        st.json(stats)
    uploaded_files = session.list_files()
    if uploaded_files:
        for f in uploaded_files:
            st.markdown(f"- {os.path.basename(f)}")
    else:
        st.info("No files in this session yet.")

# -----------------------
# Retrieve & Learn tab
# -----------------------
with tabs[1]:
    st.header("Retrieve & Learn")
    st.write("Ask a question about your uploaded materials. The system will retrieve relevant chunks, answer using only retrieved context, then optionally run a quiz and revision loop.")

    query = st.text_input("Your question:", placeholder="e.g., What is an agent function?")
    selected_topic = st.text_input("Optional: limit to a topic (subject) from your indexed docs", placeholder="e.g., Artificial Intelligence")

    if st.button("🔎 Retrieve & Answer", key="retrieve_btn"):
        if not query or not query.strip():
            st.error("Please enter a question.")
        else:
            if not pinecone_db:
                st.error("PineconeDB not configured — cannot retrieve.")
            else:
                with st.spinner("Retrieving relevant chunks..."):
                    res = pinecone_db.query(query, top_k=4, topic_filter=selected_topic or None)
                    docs = res.get("documents", [])
                    metas = res.get("metadatas", [])
                    if not docs:
                        st.warning("No relevant content found for this query. Try a different query or upload more documents.")
                        st.session_state.last_retrieval_context = ""
                    else:
                        context = "\n\n---\n\n".join(docs[:4])
                        st.session_state.last_retrieval_context = context

                        answer_prompt = f"""
You are MemoryPalAI — a retrieval-grounded assistant. Use ONLY the CONTEXT below to answer the QUESTION. If the answer is not present, reply: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{query}
"""
                        llm = get_llm("gemini-2.5-flash")
                        answer = generate_with_retry(llm, answer_prompt, retries=2)
                        st.subheader("🧠 Final Answer (based on retrieved docs)")
                        st.markdown(answer)

                        st.session_state.history.append({"query": query, "response": answer, "time": time.time()})

                        planner_prompt = f"""
You are a planning assistant. User goal: "Learn this topic better".
Context summary from retrieved materials:
{context[:1000]}

Return a concise learning plan (3 phases). Output markdown only.
"""
                        plan = generate_with_retry(llm, planner_prompt, retries=2)
                        st.subheader("🗓️ Suggested Learning Plan")
                        st.markdown(plan)

                        # --- FIX 3: CORRECTED GRAPH RENDERING LOGIC ---
                        graph_data = {"nodes": [], "edges": [], "subject": selected_topic or (metas[0].get("subject") if metas else "Unknown"), "style": metas[0].get("style") if metas else "Unknown"}
                        node_ids = set()
                        for m in metas:
                            subj = m.get("topic") or m.get("subject")
                            
                            if subj: # Only add node if subject is valid
                                if subj not in node_ids:
                                    graph_data["nodes"].append({"id": subj, "type": "Topic"})
                                    node_ids.add(subj)
                                
                                for t in (m.get("tags", []) or []):
                                    if t and t not in node_ids:
                                        graph_data["nodes"].append({"id": t, "type": "Tag"})
                                        node_ids.add(t)
                                    # Ensure edge has valid source and target
                                    if t and subj: # This check is now safe
                                        graph_data["edges"].append({"source": subj, "target": t, "label": "has_tag"})

                        if graph_data.get("nodes"):
                            st.subheader("🕸️ Knowledge Graph (lightweight)")
                            try:
                                render_knowledge_graph(graph_data)
                            except Exception as e:
                                st.warning("Graph render failed: " + str(e))
                        # --- END OF FIX 3 ---

                        st.session_state._last_plan = plan
                        st.session_state._last_subject = graph_data.get("subject", "Unknown")

    if st.session_state.last_retrieval_context:
        st.markdown("---")
        st.subheader("🔎 Retrieved Context (preview)")
        st.write(st.session_state.last_retrieval_context[:2000] + ("..." if len(st.session_state.last_retrieval_context) > 2000 else ""))

        # QUIZ generation block
        # ... (rest of the file is unchanged) ...
        st.markdown("---")
        st.subheader("🧩 Generate & Take a Quiz")

        if st.button("🧠 Generate Quiz from Retrieved Material", key="gen_quiz"):
            try:
                subject_for_quiz = st.session_state.get("_last_subject", "General")
                roadmap = st.session_state.get("_last_plan", "")
                with st.spinner("Generating quiz..."):
                    quiz_text = quiz_agent.generate_quiz(subject=subject_for_quiz, roadmap=roadmap, num_questions=4, user_goal="Improve understanding")
                    st.session_state._last_quiz = quiz_text
                    st.success("✅ Quiz generated and stored in session.")
            except Exception as e:
                st.error(f"Quiz generation failed: {e}")

        if st.session_state.get("_last_quiz"):
            st.markdown("### Quiz (interactive)")
            quiz_text = st.session_state.get("_last_quiz")
            blocks = [b.strip() for b in re.split(r"-{3,}", quiz_text) if b.strip()]
            answers_for_eval = {}

            with st.form("quiz_form"):
                for i, b in enumerate(blocks):
                    lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
                    if not lines:
                        continue
                    qline = lines[0]
                    opts = []
                    for ln in lines[1:]:
                        m = re.match(r"^[A-D]\s*[\)\.]\s*(.+)", ln)
                        if m:
                            opts.append(m.group(1).strip())
                        else:
                            if ln:
                                opts.append(ln)
                    while len(opts) < 4:
                        opts.append("N/A")
                    st.markdown(f"**{qline}**")
                    sel = st.radio(
    f"Select answer for Q{i+1}",
    options=opts,
    key=f"q_radio_{i}",
    label_visibility="collapsed"
)

                    answers_for_eval[f"Q{i+1}"] = sel
                submitted = st.form_submit_button("Submit Quiz Answers")

            if submitted:
                letter_answers = {}
                for i, b in enumerate(blocks):
                    lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
                    opts = []
                    for ln in lines[1:]:
                        m = re.match(r"^[A-D]\s*[\)\.]\s*(.+)", ln)
                        if m:
                            opts.append(m.group(1).strip())
                        else:
                            if ln:
                                opts.append(ln)
                    while len(opts) < 4:
                        opts.append("N/A")
                    chosen_text = answers_for_eval.get(f"Q{i+1}", opts[0])
                    if chosen_text in opts:
                        idx = opts.index(chosen_text)
                        letter_answers[f"Q{i+1}"] = ["A", "B", "C", "D"][idx]
                    else:
                        letter_answers[f"Q{i+1}"] = "A"

                with st.spinner("Evaluating quiz..."):
                    eval_text = quiz_agent.evaluate_answers(subject=st.session_state.get("_last_subject", "Unknown"), user_answers=letter_answers, quiz_text=quiz_text)
                    st.session_state._last_evaluation = eval_text
                    st.subheader("📊 Quiz Evaluation")
                    st.markdown(eval_text)

                    pct = None
                    try:
                        m = re.search(r"(\d+)\s*(?:out of|/)\s*(\d+)", eval_text)
                        if m:
                            score = int(m.group(1))
                            total = int(m.group(2))
                            pct = (score / total) * 100
                    except Exception:
                        pct = None

                    if pct is not None and pct < 70:
                        st.info("Low score detected — generating concise revision materials...")
                        rev = revision_agent.revise(topic=st.session_state.get("_last_subject", "Unknown"), evaluation_text=eval_text)
                        st.subheader("🔁 Revision Notes")
                        st.markdown(rev)
                    else:
                        if pct is None:
                            st.info("Could not detect score; skipping auto-revision.")
                        else:
                            st.success(f"Good performance ({pct:.1f}%) — no revision needed.")

    # Recent queries area
    st.markdown("---")
    st.header("Recent Queries")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-6:]):
            tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("time", time.time())))
            st.markdown(f"**[{tm}] ❓ {item['query']}**")
            st.caption(item['response'][:300] + "...")
            st.markdown("---")
    else:
        st.caption("No queries yet.")

st.markdown("---")
# --- FIX 4: CORRECTED THE CAPTION ---
st.caption("Built with ❤️ using Gemini, Whisper, Google Gemini Embeddings, and Pinecone.")