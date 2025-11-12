import json
import os
from app.services.llm_service import get_llm, generate_with_retry

class QuizAgent:
    """
    Generates quizzes based on subject, style, and learning roadmap.
    Evaluates user's answers and stores performance memory.
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        self.memory_file = os.path.join("app", "database", "user_profile.json")
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        print("✅ QuizAgent initialized with Gemini-2.5-flash.")

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return {}
        with open(self.memory_file, "r") as f:
            return json.load(f)

    def _save_memory(self, data):
        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)

    def generate_quiz(self, subject: str, roadmap: str = "", num_questions: int = 4, user_goal: str = ""):
        """
        Generates 3–5 multiple-choice questions from the given subject and roadmap context.
        The num_questions and user_goal arguments are optional to support flexible pipeline calls.
        """
        prompt = f"""
        You are MemoryPalAI's Quiz Agent.

        Topic: "{subject}"
        Learning Goal: "{user_goal}"

        Based on this roadmap summary:
        {roadmap[:1000]}

        Generate **{num_questions} multiple-choice questions** to test conceptual understanding.
        Each question must include exactly four options (A–D) and clearly mark the correct answer with a ✅.

        Output format example:
        Q1. What is Artificial Intelligence?
        A) A branch of physics
        B) Machines simulating human intelligence ✅
        C) Human emotions
        D) Mechanical processes
        ---
        Q2. ...
        """
        try:
            quiz_text = generate_with_retry(self.llm, prompt, retries=3)
            return quiz_text.strip() if isinstance(quiz_text, str) else str(quiz_text)
        except Exception as e:
            return f"❌ Quiz generation failed: {e}"

    def evaluate_answers(self, subject: str, user_answers: dict, quiz_text: str):
        """
        Evaluates user answers against the generated quiz content using the LLM.
        Returns a short textual evaluation summary.
        """
        prompt = f"""
        You are MemoryPalAI's Evaluator Agent.

        Evaluate the following quiz for the topic "{subject}".

        Quiz:
        {quiz_text}

        User Answers:
        {json.dumps(user_answers, indent=2)}

        Provide:
        - Correct and incorrect answers
        - Overall score (e.g., "2 out of 4 (50%)")
        - List of topics to revise
        Return as clean readable markdown text.
        """
        try:
            result = generate_with_retry(self.llm, prompt)
            text_result = result.strip() if isinstance(result, str) else str(result)

            # Save attempt history
            memory = self._load_memory()
            subject_data = memory.get(subject, {"attempts": 0, "score_history": []})
            subject_data["attempts"] += 1
            subject_data["score_history"].append(text_result)
            memory[subject] = subject_data
            self._save_memory(memory)
            return text_result
        except Exception as e:
            return f"❌ Evaluation failed: {e}"
