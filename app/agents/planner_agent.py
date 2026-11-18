# app/agents/planner_agent.py
import logging
from app.services.llm_service import get_llm, generate_with_retry

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    PlannerAgent designs customized, structured 3-phase study roadmaps
    aligned with user style and knowledge requirements.
    """

    def __init__(self):
        self.llm = get_llm()
        logger.info("PlannerAgent initialized.")

    def create_plan(self, user_goal: str, user_knowledge_summary: str, subject: str = "Unknown", style: str = "Descriptive") -> str:
        prompt = f"""
You are MemoryPalAI's Master Learning Architect.

Design an actionable, highly structured **3-Phase Mastery Roadmap** for:
- Subject: **{subject}**
- Learning Goal: **{user_goal}**
- Preferred Learning Style: **{style}**

Context from study documents:
{user_knowledge_summary[:1200]}

Roadmap Requirements:
1. Divide into 3 distinct chronological phases:
   - **Phase 1: Foundations & Core Concepts** (Estimated study time)
   - **Phase 2: Deep Dive & Practical Applications** (Estimated study time)
   - **Phase 3: Advanced Mastery & Problem Solving** (Estimated study time)
2. Include 2-3 specific learning milestones per phase.
3. At the end of each phase, provide 2 realistic YouTube search links (markdown links: [Video Title](https://www.youtube.com/results?search_query=...)).
4. Adapt the phrasing to suit the **{style}** style.

Return clean Markdown only.
"""
        try:
            return generate_with_retry(self.llm, prompt, retries=2)
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return f"❌ Roadmap generation failed: {e}"
