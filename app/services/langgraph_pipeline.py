# app/services/langgraph_pipeline.py
import os
import json
import re
import time
from typing import Optional

from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from app.agents.ingestion_agent import IngestionAgent
from app.agents.organizer_agent import OrganizerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.planner_agent import PlannerAgent

# Optional agents — pipeline will still run even if these aren't present,
# but you should implement QuizAgent/RevisionAgent for full functionality.
try:
    from app.agents.quiz_agent import QuizAgent
except Exception:
    QuizAgent = None

try:
    from app.agents.revision_agent import RevisionAgent
except Exception:
    RevisionAgent = None

from app.services.llm_service import get_llm, generate_with_retry

# ---------------- MemoryPalAI State ----------------
class MemoryPalAIState(BaseModel):
    file_path: Optional[str] = None
    query: Optional[str] = None
    user_goal: Optional[str] = None
    documents: Optional[list] = None
    graph_data: Optional[dict] = None
    retrieval_results: Optional[dict] = None
    response: Optional[str] = None
    plan: Optional[str] = None
    quiz: Optional[str] = None
    evaluation: Optional[str] = None
    revision: Optional[str] = None


# ---------------- Helpers ----------------
USER_PROFILE_PATH = os.path.join("app", "database", "user_profile.json")
os.makedirs(os.path.dirname(USER_PROFILE_PATH), exist_ok=True)


def _load_user_profile():
    if os.path.exists(USER_PROFILE_PATH):
        try:
            with open(USER_PROFILE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_user_profile(mem: dict):
    with open(USER_PROFILE_PATH, "w") as f:
        json.dump(mem, f, indent=2)


def _extract_score_from_evaluation(eval_text: str):
    """
    Try multiple patterns to extract score and total from evaluation text.
    Returns (score:int, total:int) or (None, None) if not found.
    """
    if not eval_text:
        return None, None

    # Pattern like: "Overall Score: 2 out of 4" or "Overall Score — 2/4"
    patterns = [
        r"Overall\s*Score[:\s\-\*]*\s*([\d]+)\s*(?:/|out of)\s*([\d]+)",
        r"Detected score[:\s]*([\d]+)\s*/\s*([\d]+)",
        r"Score[:\s]*([\d]+)\s*out of\s*([\d]+)",
        r"(\d+)\s*out of\s*(\d+)"
    ]
    for p in patterns:
        m = re.search(p, eval_text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                continue

    # Try fraction style "2/4"
    m = re.search(r"(\d+)\s*/\s*(\d+)", eval_text)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    return None, None


# ---------------- Graph Build ----------------
def build_memorypal_graph() -> CompiledStateGraph:
    ingestion_agent = IngestionAgent()
    organizer_agent = OrganizerAgent()
    retriever_agent = RetrieverAgent()
    planner_agent = PlannerAgent()
    llm = get_llm("gemini-2.5-flash")

    quiz_agent = QuizAgent() if QuizAgent else None
    revision_agent = RevisionAgent() if RevisionAgent else None

    graph = StateGraph(MemoryPalAIState)

    # ----- Ingest node -----
    def ingest_node(state: MemoryPalAIState):
        print(f"🔍 ingest_node received file_path={state.file_path}")
        if not state.file_path or not os.path.exists(state.file_path):
            print("❌ No valid file_path provided -> skipping ingestion.")
            state.documents = []
            return state
        docs = ingestion_agent.ingest(state.file_path)
        state.documents = docs or []
        return state

    # ----- Organize node -----
    def organize_node(state: MemoryPalAIState):
        docs = state.documents or []
        print(f"🔍 organize_node received {len(docs)} document(s)")
        if not docs:
            state.graph_data = {"nodes": [{"id": "Document", "type": "Text"}], "edges": [], "subject": "Unknown", "style": "Unknown"}
            return state

        # join small docs safely; preserve per-doc metadata if present
        text = " \n\n ".join([getattr(d, "page_content", str(d)) if hasattr(d, "page_content") else str(d) for d in docs])
        try:
            graph_data = organizer_agent.extract_graph_data(text)
            # ensure minimal structure
            if not isinstance(graph_data, dict):
                graph_data = {"nodes": [{"id": "Document", "type": "Text"}], "edges": [], "subject": "Unknown", "style": "Unknown"}
        except Exception as e:
            print(f"❌ organizer_agent.extract_graph_data error: {e}")
            graph_data = {"nodes": [{"id": "Document", "type": "Text"}], "edges": [], "subject": "Unknown", "style": "Unknown"}

        # Normalize graph_data
        graph_data.setdefault("nodes", [{"id": "Document", "type": "Text"}])
        graph_data.setdefault("edges", [])
        graph_data.setdefault("subject", "Unknown")
        graph_data.setdefault("style", "Unknown")

        state.graph_data = graph_data
        return state

    # ----- Store node -----
    def store_node(state: MemoryPalAIState):
        docs = state.documents or []
        print("🔍 store_node storing documents to vector DB...")
        if not docs:
            return state
        for i, doc in enumerate(docs):
            # doc may be LangChain Document or plain object
            content = getattr(doc, "page_content", str(doc))
            metadata = getattr(doc, "metadata", {}) or {}
            # inject subject/style if available
            if state.graph_data:
                metadata.setdefault("subject", state.graph_data.get("subject", "Unknown"))
                metadata.setdefault("style", state.graph_data.get("style", "Unknown"))
            metadata.setdefault("source", metadata.get("source", state.file_path))
            try:
                retriever_agent.add_document(f"doc_{i}", content, metadata)
            except Exception as e:
                print(f"❌ Error adding document to vector DB: {e}")
        return state

    # ----- Retrieve node -----
    def retrieve_node(state: MemoryPalAIState):
        query = state.query or ""
        if not query.strip():
            print("⚠️ No query provided — skipping retrieval.")
            state.retrieval_results = {"documents": [], "metadatas": [], "distances": []}
            return state
        print("🔍 retrieve_node running query:", query)
        try:
            res = retriever_agent.query(query, top_k=3)
            state.retrieval_results = res
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            state.retrieval_results = {"documents": [], "metadatas": [], "distances": []}
        return state

    # ----- Answer node -----
    def answer_node(state: MemoryPalAIState):
        print("🔍 answer_node generating answer...")
        results = state.retrieval_results or {}
        docs = results.get("documents", [])
        if not docs:
            state.response = "❌ No relevant documents found to answer the query."
            return state

        # Build context from top documents (first doc may be a string or list depending on VectorDB)
        top_doc = docs[0]
        if isinstance(top_doc, list):
            context = "\n\n".join(top_doc[:3])
        elif isinstance(top_doc, str):
            context = top_doc
        else:
            context = str(top_doc)

        prompt = f"""
You are MemoryPalAI, a retrieval-grounded assistant. Use only the provided context to answer.
If the answer isn't present in the context, reply: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{state.query or ''}
"""
        try:
            resp_text = generate_with_retry(llm, prompt)
            state.response = resp_text if isinstance(resp_text, str) else str(resp_text)
        except Exception as e:
            state.response = f"❌ Error generating response: {e}"
        return state

    # ----- Plan node -----
    def plan_node(state: MemoryPalAIState):
        print("🔍 plan_node creating roadmap...")
        user_goal = state.user_goal or "Learn effectively."
        # summarize retrieved docs to build knowledge summary
        retrieved_docs = (state.retrieval_results or {}).get("documents", [[""]])
        try:
            if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
                first = retrieved_docs[0]
                if isinstance(first, list):
                    knowledge_summary = " ".join(first[:3])
                else:
                    knowledge_summary = str(first)[:2000]
            else:
                knowledge_summary = ""
            plan_text = planner_agent.create_plan(user_goal, knowledge_summary)
            # planner_agent may return plain text or an object; coerce to string
            state.plan = plan_text if isinstance(plan_text, str) else str(plan_text)
        except Exception as e:
            state.plan = f"❌ Planner error: {e}"
        return state

    # ----- Quiz node -----
    def quiz_node(state: MemoryPalAIState):
        print("🧩 quiz_node generating quiz and evaluating understanding...")
        if not QuizAgent:
            state.quiz = "❌ QuizAgent not available."
            state.evaluation = ""
            return state

        try:
            # Generate quiz questions
            graph_data = state.graph_data or {}
            subject = graph_data.get("subject", "Unknown")
            roadmap = state.plan or ""
            quiz_text = quiz_agent.generate_quiz(
                subject=subject,
                roadmap=roadmap,
                num_questions=4,
                user_goal=state.user_goal or "Improve understanding"
            )
            state.quiz = quiz_text

            # --- Simulate user answers for now (can replace with actual UI input) ---
            print("\n🧠 Simulating user answers for quiz evaluation...")
            simulated_answers = {}
            for i, block in enumerate(quiz_text.split("---")):
                if not block.strip():
                    continue
                simulated_answers[f"Q{i+1}"] = "B"  # pretend user always picks option B for now

            print(f"🧩 User Answers (simulated): {simulated_answers}")

            # --- Evaluate ---
            eval_text = quiz_agent.evaluate_answers(subject, simulated_answers, quiz_text)
            state.evaluation = eval_text
            print("\n📊 Evaluation Result:\n", eval_text)

        except Exception as e:
            state.quiz = f"❌ Quiz generation error: {e}"
            state.evaluation = ""
        return state

    # ----- Revision node -----
    def revision_node(state: MemoryPalAIState):
        print("🔁 revision_node checking user performance and generating revision if needed...")
        eval_text = state.evaluation or ""
        # Try to extract a score
        score, total = _extract_score_from_evaluation(eval_text)
        if score is None or total is None or total == 0:
            print("⚠️ No valid score found in evaluation; skipping revision step.")
            state.revision = "No score detected — revision skipped."
            return state

        percentage = round((score / total) * 100, 2)
        print(f"📊 Detected score: {score}/{total} ({percentage}%)")

        # Load existing user profile
        memory = _load_user_profile()
        subject = (state.graph_data or {}).get("subject", "Unknown")

        subject_data = memory.get(subject, {"attempts": 0, "score_history": [], "avg_score": 0.0})
        subject_data["attempts"] = int(subject_data.get("attempts", 0)) + 1

        # Clean existing history to floats
        existing_history = subject_data.get("score_history", [])
        clean_history = []
        for s in existing_history:
            try:
                clean_history.append(float(s))
            except Exception:
                continue
        clean_history.append(float(percentage))
        subject_data["score_history"] = clean_history
        subject_data["avg_score"] = round(sum(clean_history) / len(clean_history), 2) if clean_history else float(percentage)

        memory[subject] = subject_data
        _save_user_profile(memory)
        print(f"📈 Updated user profile for '{subject}': {subject_data}")

        # If performance low, call revision agent to produce tailored revision
        if percentage < 70:
            if not RevisionAgent:
                state.revision = "Low performance detected but RevisionAgent not available."
            else:
                try:
                    state.revision = revision_agent.revise(topic=subject, evaluation_text=eval_text)
                except Exception as e:
                    state.revision = f"❌ RevisionAgent error: {e}"
        else:
            state.revision = "✅ Good performance — no revision needed."

        return state

    # ----- Graph wiring -----
    graph.add_node("ingest", ingest_node)
    graph.add_node("organize", organize_node)
    graph.add_node("store", store_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("plan", plan_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("revision", revision_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "organize")
    graph.add_edge("organize", "store")
    graph.add_edge("store", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "plan")
    graph.add_edge("plan", "quiz")
    graph.add_edge("quiz", "revision")
    graph.add_edge("revision", END)

    compiled_graph = graph.compile()
    print("✅ LangGraph StateGraph compiled successfully.")
    return compiled_graph


# ---------------- Test Runner ----------------
if __name__ == "__main__":
    compiled = build_memorypal_graph()

    # Create a small test file
    test_path = os.path.join(os.getcwd(), "test_note.txt")
    if not os.path.exists(test_path):
        with open(test_path, "w") as f:
            f.write("Artificial Intelligence involves Machine Learning, Deep Learning, and Neural Networks.")

    state = MemoryPalAIState(
        file_path=test_path,
        query="What is Artificial Intelligence?",
        user_goal="Learn AI in depth"
    )

    # LangGraph expects a plain dict / model_dump
    try:
        input_payload = state.model_dump() if hasattr(state, "model_dump") else state.dict()
        print("🚀 Running pipeline...")
        result = compiled.invoke(input_payload)
        if result:
            print("\n🧠 Answer:", result.get("response"))
            print("\n🗓️ Plan:", result.get("plan"))
            print("\n🕸️ Graph Data:", result.get("graph_data"))
            print("\n🧩 Quiz:", result.get("quiz"))
            print("\n📊 Evaluation:", result.get("evaluation"))
            print("\n🔁 Revision:", result.get("revision"))
        else:
            print("❌ Pipeline failed — no final state returned.")
    except Exception as e:
        print(f"❌ Pipeline invoke error: {e}")
