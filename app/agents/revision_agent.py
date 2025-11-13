# app/agents/revision_agent.py
import json
import os
from app.services.llm_service import get_llm, generate_with_retry

class RevisionAgent:
    # ... (init and _load_memory, _save_memory are the same) ...

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return {}
        with open(self.memory_file, "r") as f:
            return json.load(f)

    def _save_memory(self, data):
        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)

    # --- 1. MODIFIED SIGNATURE ---
    # Add 'style' and 'tone' parameters with defaults
    def revise(self, topic: str = None, subject: str = None, evaluation_text: str = "", style: str = "Descriptive", tone: str = "Neutral"):
        """
        Generates concise revision notes, adapting to the user's learning style.
        """
        subject = subject or topic or "Unknown Topic"
        print(f"🔁 RevisionAgent triggered for subject: {subject} (Style: {style}, Tone: {tone})")

        # --- 2. MODIFIED PROMPT ---
        # Injects the style and tone into the prompt
        prompt = f"""
        You are MemoryPalAI's Revision Agent.
        Your explanation style must be: **{style}**
        Your tone must be: **{tone}**

        The following is an evaluation of the user's quiz performance:
        {evaluation_text}

        Identify only the **top 2–3 weak subtopics**.

        For each weak topic, generate (in your specified style/tone):
        1. A short explanation (2–3 sentences maximum)
        2. One YouTube search query (keep it concise)
        3. One short self-check question

        Keep total output under 300 words.
        Format neatly in markdown like this:

        ### Topic: Example Concept
        **Explanation:** A short and clear summary (in {style} style).

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