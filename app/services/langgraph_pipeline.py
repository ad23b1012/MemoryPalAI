# app/services/langgraph_pipeline.py
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.agents.ingestion_agent import IngestionAgent
from app.agents.organizer_agent import OrganizerAgent
from app.agents.retriever_agent import RetrieverAgent
from app.agents.planner_agent import PlannerAgent
from app.services.llm_service import get_llm


# ---------------- MemoryPalAI State ----------------
class MemoryPalAIState(BaseModel):
    file_path: Optional[str] = None
    query: Optional[str] = None
    user_goal: Optional[str] = None
    documents: Optional[list] = None
    graph_data: Optional[dict] = None
    retrieval_results: Optional[dict] = None
    answer: Optional[str] = None
    plan: Optional[str] = None


# ---------------- Graph Builder ----------------
def build_memorypal_graph() -> CompiledStateGraph:
    ingestion_agent = IngestionAgent()
    organizer_agent = OrganizerAgent()
    retriever_agent = RetrieverAgent()
    planner_agent = PlannerAgent()
    llm = get_llm("gemini-2.5-flash")

    graph = StateGraph(MemoryPalAIState)

    # ----- Node: Ingestion -----
    def ingest_node(state: MemoryPalAIState):
        print(f"🔍 ingest_node received file_path={state.file_path}")
        if not state.file_path:
            print("❌ No file path in state.")
            return state
        docs = ingestion_agent.ingest(state.file_path)
        state.documents = docs
        return state

    # ----- Node: Organizer -----
    def organize_node(state: MemoryPalAIState):
        print(f"🔍 organize_node received {len(state.documents or [])} documents")
        if state.documents:
            text = " ".join([d.page_content for d in state.documents])
            graph_data = organizer_agent.extract_graph_data(text)
            state.graph_data = graph_data
        return state

    # ----- Node: Store -----
    def store_node(state: MemoryPalAIState):
        print("🔍 store_node storing documents to vector DB...")
        if state.documents:
            for i, doc in enumerate(state.documents):
                retriever_agent.add_document(f"doc_{i}", doc.page_content, doc.metadata)
        return state

    # ----- Node: Retrieve -----
    def retrieve_node(state: MemoryPalAIState):
        print("🔍 retrieve_node running query:", state.query)
        results = retriever_agent.query(state.query or "")
        state.retrieval_results = results
        return state

    # ----- Node: Answer -----
    def answer_node(state: MemoryPalAIState):
        print("🔍 answer_node generating answer...")
        results = state.retrieval_results
        if not results or not results.get("documents"):
            state.answer = "❌ No relevant information found."
            return state

        context = "\n\n".join(results["documents"][0])
        query = state.query or ""
        prompt = f"""
        You are MemoryPalAI, a personalized knowledge assistant.
        Context:
        {context}

        Question:
        {query}
        """
        response = llm.generate_content(prompt)
        state.answer = response.text.strip()
        return state

    # ----- Node: Plan -----
    def plan_node(state: MemoryPalAIState):
        print("🔍 plan_node creating roadmap...")
        user_goal = state.user_goal or "Learn effectively."
        retrieved_docs = state.retrieval_results.get("documents", [[""]])[0] if state.retrieval_results else [""]
        knowledge_summary = " ".join(retrieved_docs)
        plan = planner_agent.create_plan(user_goal, knowledge_summary)
        state.plan = plan
        return state

    # ----- Graph Construction -----
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

    compiled_graph = graph.compile()
    print("✅ LangGraph StateGraph compiled successfully.")
    return compiled_graph


# ---------------- Test Runner ----------------
if __name__ == "__main__":
    # Ensure test file exists
    test_file = "test_note.txt"
    with open(test_file, "w") as f:
        f.write("Artificial Intelligence enables machines to think and learn like humans.")

    compiled_graph = build_memorypal_graph()
    state = MemoryPalAIState(
        file_path=test_file,
        query="What is Artificial Intelligence?",
        user_goal="Learn AI in depth"
    )

    print("🚀 Running MemoryPalAI LangGraph pipeline...\n")
    result_state = compiled_graph.invoke(state)

    if result_state:
        print("\n🧠 Final Answer:\n", result_state.get("answer"))
        print("\n🗓️ Personalized Plan:\n", result_state.get("plan"))
    else:
        print("❌ Pipeline failed — no final state returned.")
