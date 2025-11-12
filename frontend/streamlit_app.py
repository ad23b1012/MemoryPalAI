import sys
import os
import tempfile
import streamlit as st

# Make backend modules importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.langgraph_pipeline import build_memorypal_graph, MemoryPalAIState
from app.agents.retriever_agent import RetrieverAgent
from app.agents.quiz_agent import QuizAgent
from app.agents.revision_agent import RevisionAgent
from components.session_manager import SessionManager
from components.graph_view import render_knowledge_graph

# -----------------------
# STREAMLIT CONFIG
# -----------------------
st.set_page_config(
    page_title="MemoryPalAI Dashboard 🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 MemoryPalAI — Interactive Learning Assistant")
st.caption("Powered by Gemini 2.5 Flash + LangGraph + Whisper + ChromaDB")

# -----------------------
# SESSION INITIALIZATION
# -----------------------
session = SessionManager()

if "graph" not in st.session_state:
    st.session_state.graph = build_memorypal_graph()

if "retriever" not in st.session_state:
    st.session_state.retriever = RetrieverAgent()

if "quiz_agent" not in st.session_state:
    st.session_state.quiz_agent = QuizAgent()

if "revision_agent" not in st.session_state:
    st.session_state.revision_agent = RevisionAgent()

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# MAIN INTERFACE
# -----------------------
col1, col2 = st.columns([1.8, 1])

# =========================================================
# LEFT PANEL – CHAT INTERFACE
# =========================================================
with col1:
    st.header("💬 Chat with MemoryPalAI")

    # Chat-like input
    uploaded_file = st.file_uploader(
        "📤 Upload a document (Text, PDF, or Audio)",
        type=["txt", "pdf", "mp3", "m4a", "wav"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        session.add_file(file_path)
        st.success(f"✅ Uploaded: `{uploaded_file.name}`")

    query = st.text_input("Ask your question:", placeholder="e.g., What is Hill Climbing in AI?")
    user_goal = st.text_input("🎯 Learning Goal:", placeholder="e.g., Understand AI search algorithms")

    if st.button("🚀 Run MemoryPalAI"):
        uploaded_files = session.list_files()

        if not uploaded_files:
            st.error("⚠️ Please upload a file first.")
        elif not query:
            st.error("⚠️ Please enter a question.")
        else:
            file_path = uploaded_files[-1]

            state = MemoryPalAIState(
                file_path=file_path,
                query=query,
                user_goal=user_goal or "Learn effectively",
            )

            with st.spinner("🤖 Processing your file and question..."):
                compiled_graph = st.session_state.graph
                result_state = compiled_graph.invoke(state.dict())

            if result_state:
                st.session_state.history.append({
                    "query": query,
                    "response": result_state.get("response", ""),
                    "plan": result_state.get("plan", ""),
                    "quiz": result_state.get("quiz", ""),
                    "evaluation": "",
                    "revision": ""
                })

                st.success("✅ MemoryPalAI completed successfully!")

                # Display response
                st.subheader("🧠 Response")
                st.markdown(result_state.get("response", "❌ No response generated."))

                # Display learning roadmap
                st.subheader("🗓️ Learning Plan")
                st.markdown(result_state.get("plan", "❌ No plan generated."))

                # Display quiz
                quiz_text = result_state.get("quiz", "")
                if quiz_text:
                    st.subheader("🧩 Take the Quiz")
                    lines = [l for l in quiz_text.split("\n") if l.strip()]
                    user_answers = {}
                    current_q = None
                    for line in lines:
                        if line.startswith("Q"):
                            current_q = line.strip()
                            st.markdown(f"**{current_q}**")
                            user_answers[current_q] = None
                        elif line.startswith(("A)", "B)", "C)", "D)")):
                            option = line.strip()
                            if current_q:
                                chosen = st.radio(
                                    f"Select answer for {current_q}",
                                    options=["A", "B", "C", "D"],
                                    key=f"{current_q}_ans"
                                )
                                user_answers[current_q] = chosen
                                break  # avoid repeating same question radios

                    if st.button("🧠 Submit Quiz"):
                        quiz_agent: QuizAgent = st.session_state.quiz_agent
                        revision_agent: RevisionAgent = st.session_state.revision_agent
                        subject = result_state.get("graph_data", {}).get("subject", query)

                        # 1️⃣ Evaluate user answers
                        try:
                            eval_text = quiz_agent.evaluate_answers(subject, user_answers, quiz_text)
                            st.markdown("### 📊 Evaluation Result")
                            st.markdown(eval_text)
                            st.session_state.history[-1]["evaluation"] = eval_text

                            # 2️⃣ Detect score and trigger revision if low
                            score_detected = 0
                            if "out of" in eval_text:
                                try:
                                    parts = eval_text.split("out of")
                                    score_detected = float(parts[0].split()[-1]) / float(parts[1].split()[0])
                                except Exception:
                                    score_detected = 0

                            if score_detected < 0.6:
                                with st.spinner("🔁 Generating personalized revision notes..."):
                                    revision_text = revision_agent.revise(subject=subject, evaluation_text=eval_text)
                                    st.markdown("### 🔁 Revision Suggestions")
                                    st.markdown(revision_text)
                                    st.session_state.history[-1]["revision"] = revision_text
                            else:
                                st.success("🎉 Great work! No revision needed this time.")
                        except Exception as e:
                            st.error(f"Quiz evaluation failed: {e}")

                # Display graph visualization
                if result_state.get("graph_data"):
                    st.subheader("🕸️ Knowledge Graph")
                    render_knowledge_graph(result_state["graph_data"])

                # Store retriever content
                st.session_state.retriever.add_document(
                    doc_id=os.path.basename(file_path),
                    content=result_state.get("response", ""),
                    metadata={"source": uploaded_file.name},
                )

            else:
                st.error("❌ Pipeline failed — no final state returned.")

# =========================================================
# RIGHT PANEL – SIDEBAR / HISTORY
# =========================================================
with col2:
    st.header("📚 Knowledge Vault")

    uploaded_files = session.list_files()
    if uploaded_files:
        st.markdown("### 📂 Uploaded Files")
        for file in uploaded_files:
            st.markdown(f"- {os.path.basename(file)}")
    else:
        st.info("No files uploaded yet.")

    st.markdown("---")
    st.header("🕓 Recent History")

    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"**❓ Query:** {item['query']}")
            st.markdown(f"**🧠 Answer:** {item['response'][:150]}...")
            if item.get("evaluation"):
                st.markdown(f"**📊 Score:** {item['evaluation'][:100]}...")
            if item.get("revision"):
                st.markdown(f"**🔁 Revision:** {item['revision'][:100]}...")
            st.markdown("---")
    else:
        st.caption("No recent queries yet.")

    if st.button("🧹 Clear All Data"):
        session.clear()
        st.session_state.history = []
        st.success("Session cleared successfully!")

st.markdown("---")
st.caption("Built with ❤️ using Gemini 2.5 Flash, LangGraph, Whisper, and ChromaDB.")
