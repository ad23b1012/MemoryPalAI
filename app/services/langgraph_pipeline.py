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
    answer: Optional[str] = None 
    plan: Optional[str] = None
    quiz: Optional[str] = None
    evaluation: Optional[str] = None
    revision: Optional[str] = None
    topic: Optional[str] = None
    document_content: Optional[str] = None

USER_PROFILE_PATH = os.path.join("app", "database", "user_profile.json")
os.makedirs(os.path.dirname(USER_PROFILE_PATH), exist_ok=True)

# ... (rest of _load_user_profile, _save_user_profile, _extract_score_from_evaluation functions are unchanged) ...
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
        if docs:
            state.document_content = " \n\n ".join([d.page_content for d in docs])
        return state

    # organize
    def organize_node(state: MemoryPalAIState):
        text = state.document_content or ""
        if not text:
            # Use the default graph from the agent
            state.graph_data = organizer_agent._default_graph() 
            return state

        try:
            # OrganizerAgent now calls style_detector internally
            graph_data = organizer_agent.extract_graph_data(text) 
            if not isinstance(graph_data, dict):
                 raise ValueError("OrganizerAgent did not return a dictionary.")
        except Exception as e:
            print(f"❌ organizer error: {e}")
            graph_data = organizer_agent._default_graph()
        
        # Ensure all keys exist
        graph_data.setdefault("nodes", [{"id":"Document","type":"Text"}])
        graph_data.setdefault("edges", [])
        graph_data.setdefault("subject", "Unknown")
        graph_data.setdefault("style", "Unknown")
        graph_data.setdefault("tone", "Neutral")
        graph_data.setdefault("tags", [])
        graph_data.setdefault("topic", graph_data.get("subject", "general"))
        
        state.graph_data = graph_data
        return state

    # store (add to vector DB)
    def store_node(state: MemoryPalAIState):
        text = state.document_content or ""
        if not text:
            return state
            
        topic = (state.graph_data or {}).get("topic", "general")
        metadata = {
            "source": state.file_path,
            "subject": (state.graph_data or {}).get("subject", "Unknown"),
            "style": (state.graph_data or {}).get("style", "Unknown"),
            "tone": (state.graph_data or {}).get("tone", "Neutral"),
            "tags": (state.graph_data or {}).get("tags", [])
        }
        
        try:
            doc_id = os.path.basename(state.file_path)
            retriever_agent.add_document(doc_id, text, metadata, topic=topic)
        except Exception as e:
            print(f"❌ Error storing document: {e}")
        return state

    # retrieve
    def retrieve_node(state: MemoryPalAIState):
        query = state.query or ""
        topic = (state.graph_data or {}).get("topic", "general")
        if not query.strip():
            state.retrieval_results = {"documents": [], "metadatas": [], "distances": []}
            return state
        try:
            # Use query, not retriever_agent.query
            res = retriever_agent.query(query_text=query, top_k=3, topic_filter=topic) 
            state.retrieval_results = res
        except Exception as e:
            print(f"❌ retrieval error: {e}")
            state.retrieval_results = {"documents": [], "metadatas": [], "distances": []}
        return state

# answer
    def answer_node(state: MemoryPalAIState):
        results = state.retrieval_results or {}
        docs = results.get("documents", [[]]) # Default to list of lists
        query = state.query or ''
        
        # --- THIS IS THE NEW, MODIFIED LOGIC ---
        
        if not docs or not docs[0]:
            # --- USE CASE 1: NO CONTEXT FOUND (FALLBACK) ---
            print("⚠️ No context found in Pinecone. Generating a general answer.")
            
            # This is the disclaimer you wanted
            disclaimer = "The provided documents did not contain an answer to this question. Here is a general answer:\n\n"
            
            # Create a general Q&A prompt (no context)
            prompt = f"Provide a general, helpful answer to the following question: {query}"
            
            try:
                resp_text = generate_with_retry(llm, prompt)
                # Combine the disclaimer and the answer
                state.answer = disclaimer + (resp_text if isinstance(resp_text, str) else str(resp_text))
            except Exception as e:
                state.answer = f"❌ Error generating general response: {e}"
                
        else:
            # --- USE CASE 2: CONTEXT FOUND (STANDARD RAG) ---
            print("✅ Context found. Generating retrieval-grounded answer.")
            
            context_list = [chunk for sublist in docs for chunk in (sublist if isinstance(sublist, list) else [sublist])]
            context = "\n\n".join(context_list[:3])

            prompt = f"""
            You are MemoryPalAI, a retrieval-grounded assistant. Use only the provided context to answer.
            If the answer isn't present in the context, reply: "I don't know based on the provided documents."

            CONTEXT:
            {context}

            QUESTION:
            {query}
            """
            try:
                resp_text = generate_with_retry(llm, prompt)
                state.answer = resp_text if isinstance(resp_text, str) else str(resp_text)
            except Exception as e:
                state.answer = f"❌ Error generating RAG response: {e}"
        
        return state
    
    # plan
    def plan_node(state: MemoryPalAIState):
        user_goal = state.user_goal or "Learn effectively."
        retrieved_docs = (state.retrieval_results or {}).get("documents", [])
        
        try:
            first_chunk = retrieved_docs[0] if retrieved_docs else ""
            knowledge_summary = str(first_chunk)[:2000]
            
            graph_data = state.graph_data or {}
            subject = graph_data.get("subject", "Unknown")
            style = graph_data.get("style", "Unknown")
            
            plan_text = planner_agent.create_plan(user_goal, knowledge_summary, subject=subject, style=style)
            state.plan = plan_text if isinstance(plan_text, str) else str(plan_text)
        except Exception as e:
            state.plan = f"❌ Planner error: {e}"
        return state

    # --- NEW: CONDITIONAL ROUTER ---
    def should_plan(state: MemoryPalAIState):
        """
        Check if a user_goal was provided.
        If yes, go to 'plan'. If no, go straight to END.
        """
        print("🚦 Checking if a goal was provided...")
        if state.user_goal and state.user_goal != "Learn effectively":
            print("✅ Goal found. Proceeding to plan_node.")
            return "plan"
        else:
            print("🛑 No specific goal. Skipping plan and quiz.")
            return "end"

    # quiz node
    def quiz_node(state: MemoryPalAIState):
        if not quiz_agent:
            state.quiz = "❌ QuizAgent not available."
            state.evaluation = ""
            return state
        try:
            graph_data = state.graph_data or {}
            subject = graph_data.get("subject", "Unknown")
            roadmap = state.plan or ""
            quiz_text = quiz_agent.generate_quiz(subject=subject, roadmap=roadmap, num_questions=4, user_goal=state.user_goal or "Improve understanding")
            state.quiz = quiz_text
            state.evaluation = state.evaluation or ""
        except Exception as e:
            state.quiz = f"❌ Quiz generation error: {e}"
            state.evaluation = ""
        return state

    # revision node
    def revision_node(state: MemoryPalAIState):
        eval_text = state.evaluation or ""
        score, total = _extract_score_from_evaluation(eval_text)
        graph_data = state.graph_data or {}
        subject = graph_data.get("subject", "Unknown")

        if score is None or total is None or total == 0:
            state.revision = "No score detected — revision skipped."
            return state
            
        percentage = round((score / total) * 100, 2)
        memory = _load_user_profile()
        subject_data = memory.get(subject, {"attempts": 0, "score_history": [], "avg_score": 0.0})
        subject_data["attempts"] = int(subject_data.get("attempts", 0)) + 1
        existing_history = subject_data.get("score_history", [])
        clean_history = [float(s) for s in existing_history if isinstance(s, (int, float))]
        clean_history.append(float(percentage))
        subject_data["score_history"] = clean_history
        subject_data["avg_score"] = round(sum(clean_history) / len(clean_history), 2)
        memory[subject] = subject_data
        _save_user_profile(memory)
        
        if percentage < 70:
            if not revision_agent:
                state.revision = "Low performance detected but RevisionAgent not available."
            else:
                try:
                    # --- INTEGRATE STYLE DETECTOR INFO ---
                    style = graph_data.get("style", "Descriptive")
                    tone = graph_data.get("tone", "Neutral")
                    
                    state.revision = revision_agent.revise(
                        subject=subject, 
                        evaluation_text=eval_text,
                        style=style,
                        tone=tone
                    )
                    # --- END OF INTEGRATION ---
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
    
    # --- USE CONDITIONAL ROUTER ---
    graph.add_conditional_edges(
        "answer", # After answering...
        should_plan, # Call this routing function
        {
            "plan": "plan", # If it returns "plan", go to plan_node
            "end": END      # If it returns "end", stop
        }
    )
    
    graph.add_edge("plan", "quiz")
    graph.add_edge("quiz", "revision")
    graph.add_edge("revision", END)

    compiled_graph = graph.compile()
    print("✅ LangGraph StateGraph compiled successfully.")
    return compiled_graph


# ---------------- Test Runner ----------------
# if __name__ == "__main__":
#     # Ensure test file exists
#     test_file = "test_note.txt"
#     with open(test_file, "w") as f:
#         f.write("Artificial Intelligence enables machines to think and learn like humans.")

#     compiled_graph = build_memorypal_graph()
    
#     print("🚀 Running pipeline WITH a specific goal...\n")
#     state_with_goal = MemoryPalAIState(
#         file_path=test_file,
#         query="What is Artificial Intelligence?",
#         user_goal="Learn AI in depth", # This is a real goal
#         evaluation="Q1: Correct\nFinal Score: 1 / 1\nTopics to revise: []" # Simulate a good score
#     )

#     result_state_with_goal = compiled_graph.invoke(state_with_goal)

#     if result_state_with_goal:
#         print("\n--- [With Goal Result] ---")
#         print(result_state_with_goal)
#         print("---------------------------\n")
#         print("\n🧠 Final Answer:\n", result_state_with_goal.get("answer")) 
#         print("\n🗓️ Personalized Plan:\n", result_state_with_goal.get("plan"))
#         print("\n🧩 Quiz:\n", result_state_with_goal.get("quiz"))
#         print("\n🔁 Revision:\n", result_state_with_goal.get("revision"))
#     else:
#         print("❌ Pipeline failed — no final state returned.")