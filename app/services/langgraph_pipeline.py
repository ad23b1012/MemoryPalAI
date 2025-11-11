# app/services/langgraph_pipeline.py
import os
from typing import Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from app.agents.ingestion_agent import IngestionAgent
from app.agents.organizer_agent import OrganizerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.planner_agent import PlannerAgent
from app.services.llm_service import get_llm, generate_with_retry

# ---------------- State model ----------------
class MemoryPalAIState(BaseModel):
    file_path: Optional[str] = None
    query: Optional[str] = None
    user_goal: Optional[str] = None
    documents: Optional[list] = None
    graph_data: Optional[dict] = None
    retrieval_results: Optional[dict] = None
    answer: Optional[str] = None
    plan: Optional[str] = None


# ---------------- Graph builder ----------------
def build_memorypal_graph() -> CompiledStateGraph:
    ingestion_agent = IngestionAgent()
    organizer_agent = OrganizerAgent()
    retriever_agent = RetrieverAgent()
    planner_agent = PlannerAgent()
    llm = get_llm("gemini-2.5-flash")

    graph = StateGraph(MemoryPalAIState)

    # ingest node
    def ingest_node(state: MemoryPalAIState):
        print(f"🔍 ingest_node received file_path={state.file_path}")
        if not state.file_path or not os.path.exists(state.file_path):
            print("❌ Invalid or missing file path.")
            state.documents = []
            return state
        docs = ingestion_agent.ingest(state.file_path)
        state.documents = docs
        return state

    # organize node
    def organize_node(state: MemoryPalAIState):
        docs = state.documents or []
        print(f"🔍 organize_node received {len(docs)} documents")
        if docs:
            text = " ".join([d.page_content for d in docs])
            graph_data = organizer_agent.extract_graph_data(text)
            state.graph_data = graph_data
        return state

    # store node
    def store_node(state: MemoryPalAIState):
        docs = state.documents or []
        print("🔍 store_node storing documents to vector DB...")
        if docs:
            for i, doc in enumerate(docs):
                # ensure metadata dict exists
                metadata = getattr(doc, "metadata", {}) or {}
                metadata.setdefault("source", state.file_path or "unknown")
                retriever_agent.add_document(f"doc_{i}", doc.page_content, metadata)
        return state

    # retrieve node
    def retrieve_node(state: MemoryPalAIState):
        q = state.query or ""
        print(f"🔍 retrieve_node running query: {q}")
        results = retriever_agent.query(q, top_k=3)
        state.retrieval_results = results
        return state

    # answer node
    def answer_node(state: MemoryPalAIState):
        print("🔍 answer_node generating answer...")
        results = state.retrieval_results or {}
        docs = results.get("documents", [])
        if not docs:
            state.answer = "❌ No relevant documents found to answer the query."
            return state

        # context assembly: docs may be strings or lists — handle both
        if isinstance(docs[0], list):
            context = "\n\n".join(docs[0][:3])
        else:
            context = "\n\n".join(docs[:3])

        prompt = f"""
You are MemoryPalAI, a retrieval-grounded assistant.
Answer only from the CONTEXT below. If the answer is not present, say: "I don't know based on the provided documents."

CONTEXT:
{context}

QUESTION:
{state.query or ''}
"""
        try:
            text = generate_with_retry(llm, prompt, generation_config={"temperature": 0.0, "max_output_tokens": 400})
            state.answer = text
        except Exception as e:
            state.answer = f"❌ Error generating answer: {e}"
        return state

    # plan node
    def plan_node(state: MemoryPalAIState):
        print("🔍 plan_node creating roadmap...")
        user_goal = state.user_goal or "Learn effectively."
        retrieved_docs = state.retrieval_results.get("documents", []) if state.retrieval_results else []
        if retrieved_docs:
            if isinstance(retrieved_docs[0], list):
                knowledge_summary = " ".join(retrieved_docs[0][:5])
            else:
                knowledge_summary = " ".join(retrieved_docs[:5])
        else:
            knowledge_summary = ""

        plan_text = planner_agent.create_plan(user_goal, knowledge_summary)
        state.plan = plan_text
        return state

    # add nodes and edges
    graph.add_node("ingest", ingest_node)
    graph.add_node("organize", organize_node)
    graph.add_node("store", store_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("plan", plan_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "organize")
    graph.add_edge("organize", "store")
    graph.add_edge("store", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "plan")
    graph.add_edge("plan", END)

    compiled = graph.compile()
    print("✅ LangGraph StateGraph compiled successfully.")
    return compiled


# ---------------- test runner ----------------
if __name__ == "__main__":
    compiled_graph = build_memorypal_graph()
    test_file = "test_note.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("Artificial Intelligence enables machines to think and learn like humans.")
    state = MemoryPalAIState(file_path=test_file, query="What is Artificial Intelligence?", user_goal="Learn AI")
    print("🚀 Running pipeline...")
    result = compiled_graph.invoke(dict(state))
    if result:
        print("\n🧠 Answer:\n", result.get("answer"))
        print("\n🗓️ Plan:\n", result.get("plan"))
        print("\n📊 Graph Data:\n", result.get("graph_data"))
    else:
        print("❌ No final state returned.")
