# app/agents/planner_agent.py
from app.services.llm_service import get_llm


class PlannerAgent:
    """
    Uses Gemini-2.5-flash to generate learning or career roadmaps.
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ PlannerAgent initialized with Gemini-2.5-flash.")

    def create_plan(self, user_goal: str, user_knowledge_summary: str):
        print("🚀 PlannerAgent: generating personalized roadmap...")
        prompt = f"""
        You are MemoryPalAI's Planning Agent.

        Generate a realistic, actionable roadmap based on the user's goal and knowledge.
        Output in clear Markdown with numbered phases and bullet points.

        ---
        Goal: {user_goal}
        Current Knowledge Summary: {user_knowledge_summary}
        ---
        """
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"❌ PlannerAgent error: {e}"
