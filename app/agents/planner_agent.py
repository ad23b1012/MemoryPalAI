# app/agents/planner_agent.py
from app.services.llm_service import get_llm, generate_with_retry

class PlannerAgent:
    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ PlannerAgent initialized with Gemini-2.5-flash.")

    def create_plan(self, user_goal: str, user_knowledge_summary: str, subject: str = "Unknown", style: str = "Unknown"):
        prompt = f"""
        You are MemoryPalAI's Planning Agent.

        Based on the given context, generate a **personalized learning roadmap** with:
        - Clear, numbered phases.
        - Learning objectives for each phase.
        - Study activities and suggestions.
        - At the end of each phase, recommend **2–3 YouTube videos**
          using realistic YouTube search links.
        - Adapt explanations to match the user's learning style.

        Example format:

        ### Phase 1: Foundations of AI
        - Learn what Artificial Intelligence means.
        - Study key subfields (Machine Learning, Deep Learning).
        🎥 Suggested Videos:
        • [Intro to AI](https://www.youtube.com/results?search_query=introduction+to+artificial+intelligence)
        • [AI Explained for Beginners](https://www.youtube.com/results?search_query=AI+explained+for+beginners)

        ---
        Goal: {user_goal}
        Current Knowledge Summary: {user_knowledge_summary[:500]}
        Subject: {subject}
        Learning Style: {style}

        Output only Markdown text. Do not include JSON or code fences.
        """
        try:
            return generate_with_retry(self.llm, prompt, retries=3)
        except Exception as e:
            return f"❌ PlannerAgent error: {e}"
