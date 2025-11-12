# app/services/langgraph_pipeline.py
import os
import json
import re
from typing import Optional

from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from app.agents.ingestion_agent import IngestionAgent
from app.agents.organizer_agent import OrganizerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.planner_agent import PlannerAgent

# Optional agents
try:
    from app.agents.quiz_agent import QuizAgent
except Exception:
    QuizAgent = None

try:
    from app.agents.revision_agent import RevisionAgent
except Exception:
    RevisionAgent = None

from app.services.llm_service import get_llm, generate_with_retry

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
    topic: Optional[str] = None

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
    if not eval_text:
        return None, None
    patterns = [
        r"(\d+)\s*out of\s*(\d+)",
        r"Overall\s*Score[:\s\-\*]*\s*([\d]+)\s*(?:/|out of)\s*([\d]+)",
        r"Score[:\s]*([\d]+)\s*out of\s*([\d]+)",
        r"Detected score[:\s]*([\d]+)\s*/\s*([\d]+)"
    ]
    for p in patterns:
        m = re.search(p, eval_text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                continue
    m = re.search(r"(\d+)\s*/\s*(\d+)", eval_text)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
    return None, None

def build_memorypal_graph() -> CompiledStateGraph:
    ingestion_agent = IngestionAgent()
    organizer_agent = OrganizerAgent()
    retriever_agent = RetrieverAgent()
    planner_agent = PlannerAgent()
    llm = get_llm("gemini-2.5-flash")

    quiz_agent = QuizAgent() if QuizAgent else None
    revision_agent = RevisionAgent() if RevisionAgent else None

    graph = StateGraph(MemoryPalAIState)

    # ingest
    def ingest_node(state: MemoryPalAIState):
        print(f"🔍 ingest_node file_path={state.file_path}")
        if not state.file_path or not os.path.exists(state.file_path):
            state.documents = []
            return state
        docs = ingestion_agent.ingest(state.file_path)
        state.documents = docs or []
        return state

    # organize
    def organize_node(state: MemoryPalAIState):
        docs = state.documents or []
        if not docs:
            state.graph_data = {"nodes":[{"id":"Document","type":"Text"}], "edges": [], "subject": "Unknown", "style": "Unknown", "topic": "general"}
            return state
        text = " \n\n ".join([getattr(d, "page_content", str(d)) if hasattr(d, "page_content") else str(d) for d in docs])
        try:
            graph_data = organizer_agent.extract_graph_data(text)
            if not isinstance(graph_data, dict):
                graph_data = {"nodes":[{"id":"Document","type":"Text"}], "edges": [], "subject": "Unknown", "style": "Unknown", "topic": "general"}
        except Exception as e:
            print("❌ organizer error:", e)
            graph_data = {"nodes":[{"id":"Document","type":"Text"}], "edges": [], "subject": "Unknown", "style": "Unknown", "topic": "general"}
        graph_data.setdefault("nodes", [{"id":"Document","type":"Text"}])
        graph_data.setdefault("edges", [])
        graph_data.setdefault("subject", "Unknown")
        graph_data.setdefault("style", "Unknown")
        graph_data.setdefault("topic", graph_data.get("subject", "general"))
        state.graph_data = graph_data
        return state

    # store (add to vector DB)
    def store_node(state: MemoryPalAIState):
        docs = state.documents or []
        if not docs:
            return state
        topic = (state.graph_data or {}).get("topic", "general")
        for i, doc in enumerate(docs):
            content = getattr(doc, "page_content", str(doc))
            metadata = getattr(doc, "metadata", {}) or {}
            metadata.setdefault("source", getattr(doc, "metadata", {}).get("source", state.file_path))
            try:
                retriever_agent.add_document(f"{os.path.basename(state.file_path)}_doc{i}", content, metadata, topic=topic)
            except Exception as e:
                print("❌ Error storing document:", e)
        return state

    # retrieve
    def retrieve_node(state: MemoryPalAIState):
        query = state.query or ""
        if not query.strip():
            state.retrieval_results = {"documents": [], "metadatas": [], "distances": []}
            return state
        try:
            res = retriever_agent.query(query, top_k=3, topic=state.topic)
            state.retrieval_results = res
        except Exception as e:
            print("❌ retrieval error:", e)
            state.retrieval_results = {"documents": [], "metadatas": [], "distances": []}
        return state

    # answer
    def answer_node(state: MemoryPalAIState):
        results = state.retrieval_results or {}
        docs = results.get("documents", [])
        if not docs:
            state.response = "I don't know based on the provided documents."
            return state
        top_doc = docs[0]
        context = "\n\n".join(docs[:3]) if isinstance(docs, list) else str(top_doc)
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

    # plan
    def plan_node(state: MemoryPalAIState):
        user_goal = state.user_goal or "Learn effectively."
        retrieved_docs = (state.retrieval_results or {}).get("documents", [""])
        try:
            first = retrieved_docs[0] if retrieved_docs else ""
            knowledge_summary = str(first)[:2000]
            plan_text = planner_agent.create_plan(user_goal, knowledge_summary, subject=(state.graph_data or {}).get("subject", "Unknown"), style=(state.graph_data or {}).get("style", "Unknown"))
            state.plan = plan_text if isinstance(plan_text, str) else str(plan_text)
        except Exception as e:
            state.plan = f"❌ Planner error: {e}"
        return state

    # quiz node — GENERATE QUIZ but DO NOT EVALUATE (UI must collect answers)
    def quiz_node(state: MemoryPalAIState):
        if not QuizAgent:
            state.quiz = "❌ QuizAgent not available."
            state.evaluation = ""
            return state
        try:
            graph_data = state.graph_data or {}
            subject = graph_data.get("subject", "Unknown")
            roadmap = state.plan or ""
            quiz_text = quiz_agent.generate_quiz(subject=subject, roadmap=roadmap, num_questions=4, user_goal=state.user_goal or "Improve understanding")
            state.quiz = quiz_text
            # Important: DO NOT simulate answers here. UI should collect answers and call evaluate() later.
            state.evaluation = state.evaluation or ""
        except Exception as e:
            state.quiz = f"❌ Quiz generation error: {e}"
            state.evaluation = ""
        return state

    # revision node — run only if evaluation text already present (UI will set state.evaluation after quiz)
    def revision_node(state: MemoryPalAIState):
        eval_text = state.evaluation or ""
        score, total = _extract_score_from_evaluation(eval_text)
        if score is None or total is None or total == 0:
            state.revision = "No score detected — revision skipped."
            return state
        percentage = round((score / total) * 100, 2)
        memory = _load_user_profile()
        subject = (state.graph_data or {}).get("subject", "Unknown")
        subject_data = memory.get(subject, {"attempts": 0, "score_history": [], "avg_score": 0.0})
        subject_data["attempts"] = int(subject_data.get("attempts", 0)) + 1
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
