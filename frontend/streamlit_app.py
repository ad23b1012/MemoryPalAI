import sys
import os
import tempfile
import streamlit as st

# Ensure backend modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.langgraph_pipeline import build_memorypal_graph, MemoryPalAIState
from app.agents.retriever_agent import RetrieverAgent
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

st.title("🧠 MemoryPalAI — Intelligent Adaptive Learning System")
st.markdown(
    "Upload documents, extract structured knowledge, and get a **personalized adaptive learning plan** with quizzes, all powered by **Gemini 2.5 Flash + LangGraph + Whisper + ChromaDB**."
)

# -----------------------
# SESSION INITIALIZATION
# -----------------------
session = SessionManager()

if "graph" not in st.session_state:
    st.session_state.graph = build_memorypal_graph()

if "retriever" not in st.session_state:
    st.session_state.retriever = RetrieverAgent()

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# MAIN INTERFACE
# -----------------------
col1, col2 = st.columns([1.7, 1])

with col1:
    st.header("📤 Upload a File")
    uploaded_file = st.file_uploader(
        "Choose a text, PDF, or audio file",
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

    # -----------------------
    # QUERY INPUT SECTION
    # -----------------------
    st.header("💬 Ask a Question / Define Your Goal")
    query = st.text_input("Your question:", placeholder="e.g., What is Artificial Intelligence?")
    user_goal = st.text_input("🎯 Your learning goal:", placeholder="e.g., Learn AI in depth")

    if st.button("🚀 Run MemoryPalAI"):
        uploaded_files = session.list_files()

        if not uploaded_files:
            st.error("⚠️ Please upload a file first.")
        elif not query:
            st.error("⚠️ Please enter a question.")
        else:
            file_path = uploaded_files[-1]

            # ✅ Create initial LangGraph state
            state = MemoryPalAIState(
                file_path=file_path,
                query=query,
                user_goal=user_goal or "Learn effectively"
            )

            with st.spinner("🤖 Processing your file and generating roadmap..."):
                compiled_graph = st.session_state.graph
                result_state = compiled_graph.invoke(state)

            if result_state:
                st.session_state.history.append({
                    "query": query,
                    "response": result_state.get("answer", "❌ No response generated."),
                    "plan": result_state.get("plan", "❌ No plan generated."),
                    "quiz": result_state.get("quiz", []),
                    "performance": result_state.get("performance", {}),
                    "adaptive_plan": result_state.get("adaptive_plan", "❌ No adaptive plan generated.")
                })

                st.success("✅ MemoryPalAI pipeline completed successfully!")

                # 🧠 Final Answer
                st.subheader("🧠 Final Answer")
                st.markdown(result_state.get("answer", "❌ No response generated."))

                # 🗓️ Personalized Learning Roadmap
                st.subheader("🗓️ Personalized Learning Roadmap")
                st.markdown(result_state.get("plan", "❌ No plan generated."))

                # 🕸 Extracted Knowledge Graph
                if result_state.graph_data:
                    st.subheader("🕸️ Extracted Knowledge Graph")
                    render_knowledge_graph(result_state.graph_data)

                # 🧩 Store in Retriever DB for future queries
                st.session_state.retriever.add_document(
                    doc_id=os.path.basename(file_path),
                    content=result_state.get("answer", ""),
                    metadata={"source": uploaded_file.name},
                )

                # -----------------------
                # 🧠 Adaptive Quiz Section
                # -----------------------
                quiz = result_state.get("quiz", [])
                if quiz:
                    st.markdown("---")
                    st.subheader("🧩 Adaptive Quiz — Test Your Understanding")

                    user_answers = {}
                    for idx, q in enumerate(quiz, start=1):
                        st.markdown(f"**Q{idx}. {q['question']}**")
                        user_answers[str(idx)] = st.radio(
                            f"Select answer for Question {idx}",
                            q["options"],
                            key=f"quiz_q{idx}"
                        )

                    if st.button("✅ Submit Answers"):
                        from app.agents.quiz_agent import QuizAgent
                        quiz_agent = QuizAgent()

                        performance = quiz_agent.evaluate_answers(quiz, user_answers)
                        adaptive_plan = quiz_agent.recommend_next_steps(
                            performance,
                            user_goal or "Learn effectively",
                            result_state.get("plan", "")
                        )

                        st.markdown("### 📊 Your Quiz Performance")
                        st.json(performance)

                        st.markdown("### 🔁 Adaptive Learning Plan (Next Steps)")
                        st.markdown(adaptive_plan)

                else:
                    st.warning("⚠️ No quiz generated for this content.")

            else:
                st.error("❌ Pipeline failed — no final state returned.")

# -----------------------
# SIDEBAR / RIGHT PANEL
# -----------------------
with col2:
    st.header("📚 Knowledge Vault")

    uploaded_files = session.list_files()
    if uploaded_files:
        for file in uploaded_files:
            st.markdown(f"- {os.path.basename(file)}")
    else:
        st.info("No files uploaded yet.")

    st.markdown("---")
    st.header("🕓 Recent Queries")

    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"**❓ Query:** {item['query']}")
            st.markdown(f"**🧠 Response:** {item['response'][:200]}...")
            st.markdown(f"**📊 Accuracy:** {item.get('performance', {}).get('accuracy', 'N/A')}%")
            st.markdown("---")
    else:
        st.caption("No recent queries yet.")

    if st.button("🧹 Clear All Data"):
        session.clear()
        st.session_state.history = []
        st.success("Session cleared successfully!")

st.markdown("---")
st.caption("Built with ❤️ using Gemini 2.5 Flash, LangGraph, Whisper, and ChromaDB.")
