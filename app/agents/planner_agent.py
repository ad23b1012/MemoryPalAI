# app/agents/planner_agent.py
from app.services.llm_service import get_llm, generate_with_retry


class PlannerAgent:
    """
    Uses Gemini-2.5-flash to generate personalized learning or career roadmaps.
    Automatically returns clean text (not object references).
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ PlannerAgent initialized with Gemini-2.5-flash.")

    def create_plan(self, user_goal: str, user_knowledge_summary: str) -> str:
        print("🚀 PlannerAgent: generating personalized roadmap...")
        prompt = f"""
You are MemoryPalAI's Planning Agent.

Based on the user's current knowledge and goal, generate a realistic and structured roadmap.
The roadmap must be clear, practical, and motivational, with step-by-step learning paths.

---
User Goal: {user_goal}
Current Knowledge Summary: {user_knowledge_summary}
---

Format:
### Personalized Learning Roadmap
1. Phase 1 – Foundations: ...
2. Phase 2 – Intermediate: ...
3. Phase 3 – Advanced: ...
4. Phase 4 – Practice & Revision: ...
"""

        try:
            text = generate_with_retry(
                self.llm,
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 800}
            )

            # ✅ make sure we return only clean text
            if not isinstance(text, str):
                try:
                    text = text.text
                except Exception:
                    text = str(text)

            return text.strip()

        except Exception as e:
            return f"❌ PlannerAgent error: {e}"


# -------------- Test --------------
if __name__ == "__main__":
    agent = PlannerAgent()
    result = agent.create_plan(
        "Learn Artificial Intelligence",
        "User has a basic understanding of AI concepts such as Machine Learning and Neural Networks."
    )
    print("\n✅ Generated Plan:\n")
    print(result)
