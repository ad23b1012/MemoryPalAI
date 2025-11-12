# app/agents/revision_agent.py
import json
import os
from app.services.llm_service import get_llm, generate_with_retry

class RevisionAgent:
    """
    Automatically revises weak topics based on quiz results and user performance memory.
    Generates simplified explanations, reinforcement questions, and learning resources.
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        self.memory_file = os.path.join("app", "database", "user_profile.json")
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        print("✅ RevisionAgent initialized with Gemini-2.5-flash.")

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return {}
        with open(self.memory_file, "r") as f:
            return json.load(f)

    def _save_memory(self, data):
        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)

    def revise(self, topic: str = None, subject: str = None, evaluation_text: str = ""):
        """
        Generates concise revision notes and brief YouTube recommendations for weak areas.
        Either 'topic' or 'subject' may be provided.
        """
        subject = subject or topic or "Unknown Topic"
        print(f"🔁 RevisionAgent triggered for subject: {subject}")

        prompt = f"""
        You are MemoryPalAI's Revision Agent.

        The following is an evaluation of the user's quiz performance:
        {evaluation_text}

        Identify only the **top 2–3 weak subtopics**.

        For each weak topic, generate:
        1. A short explanation (2–3 sentences maximum)
        2. One YouTube search query (keep it concise)
        3. One short self-check question

        Keep total output under 300 words.
        Format neatly in markdown like this:

        ### Topic: Example Concept
        **Explanation:** A short and clear summary.

        **YouTube:** https://www.youtube.com/results?search_query=example+concept+explained  
        **Self-Check:** What's one real-world use of this concept?
        """

        try:
            revision_output = generate_with_retry(self.llm, prompt, retries=3)
            revision_text = revision_output if isinstance(revision_output, str) else str(revision_output)

            memory = self._load_memory()
            subject_data = memory.get(subject)
            if not isinstance(subject_data, dict):
                subject_data = {"attempts": 0, "score_history": [], "revisions": []}
            subject_data.setdefault("revisions", [])
            subject_data["revisions"].append(revision_text)
            memory[subject] = subject_data
            self._save_memory(memory)

            print(f"✅ Concise revision generated and saved for '{subject}'")
            return revision_text

        except Exception as e:
            return f"❌ Revision generation failed: {e}"
