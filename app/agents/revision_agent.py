# app/agents/revision_agent.py
import json
import os
import re
import logging
from typing import Dict, Any, Optional
from app.services.llm_service import get_llm, generate_with_retry
from app.utils.scheduler import calculate_next_review

logger = logging.getLogger(__name__)


class RevisionAgent:
    """
    Automatically revises weak topics based on quiz results and user performance memory.
    Generates simplified explanations, reinforcement questions, YouTube resources,
    and calculates spaced repetition review schedules.
    """

    def __init__(self):
        self.llm = get_llm()
        self.memory_file = os.path.join(os.path.dirname(__file__), "..", "database", "user_profile.json")
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        logger.info("RevisionAgent initialized.")

    def _load_memory(self) -> Dict[str, Any]:
        """Loads user performance memory from JSON file with data healing."""
        if not os.path.exists(self.memory_file):
            return {}
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {}
                return data
        except Exception as e:
            logger.warning(f"Error loading user profile: {e}")
            return {}

    def _save_memory(self, data: Dict[str, Any]):
        """Saves user performance memory safely."""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")

    def record_quiz_result(self, subject: str, score_pct: float, evaluation_text: str = ""):
        """Records a quiz attempt, calculates spaced repetition interval, and updates user memory."""
        memory = self._load_memory()
        subject = subject or "General"
        subject_data = memory.get(subject, {})
        if not isinstance(subject_data, dict):
            subject_data = {}

        attempts = subject_data.get("attempts", 0) + 1
        raw_history = subject_data.get("score_history", [])
        # Sanitize score history to only numbers
        clean_history = [float(s) for s in raw_history if isinstance(s, (int, float))]
        clean_history.append(round(score_pct, 1))

        avg_score = round(sum(clean_history) / len(clean_history), 1) if clean_history else score_pct
        current_streak = subject_data.get("review_schedule", {}).get("new_streak", 0)
        review_schedule = calculate_next_review(score_pct, current_streak)

        subject_data["attempts"] = attempts
        subject_data["score_history"] = clean_history
        subject_data["avg_score"] = avg_score
        subject_data["last_score"] = round(score_pct, 1)
        subject_data["review_schedule"] = review_schedule

        memory[subject] = subject_data
        self._save_memory(memory)
        return review_schedule

    def revise(self, topic: Optional[str] = None, subject: Optional[str] = None, evaluation_text: str = "", style: str = "Descriptive", tone: str = "Neutral") -> str:
        """
        Generates concise revision notes adapting to user's style and weaknesses.
        """
        target_subject = subject or topic or "General Concepts"
        logger.info(f"RevisionAgent triggered for subject: {target_subject} (Style: {style}, Tone: {tone})")

        prompt = f"""
You are MemoryPalAI's Master Revision Agent.
Your explanation style: **{style}**
Your tone: **{tone}**

Here is the user's quiz evaluation:
{evaluation_text}

Identify the top 2-3 weak areas that need reinforcement.
For each weak topic, generate:
1. 💡 **Core Concept:** 2-3 sentence crystal-clear explanation (in {style} style).
2. 🎥 **Curated Search:** Direct YouTube search link (format: [Topic Title](https://www.youtube.com/results?search_query=...))
3. ⚡ **Self-Check Flashcard:** 1 thought-provoking recall question.

Format clearly in Markdown:

### 🔁 Revision Pack: {target_subject}

#### 1. [Concept Name]
**Explanation:** ...
**Recommended Video:** [Watch Tutorial](https://www.youtube.com/results?search_query=...)
**Active Recall Question:** ...
"""
        try:
            revision_output = generate_with_retry(self.llm, prompt, retries=2)
            revision_text = str(revision_output).strip()

            memory = self._load_memory()
            subject_data = memory.get(target_subject, {})
            if not isinstance(subject_data, dict):
                subject_data = {"attempts": 0, "score_history": [], "revisions": []}
            subject_data.setdefault("revisions", [])
            subject_data["revisions"].append(revision_text)
            memory[target_subject] = subject_data
            self._save_memory(memory)

            return revision_text
        except Exception as e:
            logger.error(f"Revision generation failed: {e}")
            return f"❌ Revision generation failed: {e}"
