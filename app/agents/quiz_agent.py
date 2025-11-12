# app/agents/quiz_agent.py

from app.services.llm_service import get_llm, generate_with_retry

class QuizAgent:
    """
    QuizAgent handles automatic quiz generation, evaluation, and feedback using Gemini.
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ QuizAgent initialized with Gemini-2.5-flash.")

    def generate_quiz(self, subject: str, roadmap: str = "", num_questions: int = 5, user_goal: str = "") -> str:
        """
        Generates MCQ-style quiz questions based on the given subject and context.
        Output format:
        Q1. Question
        A) ...
        B) ...
        C) ...
        D) ...
        """
        prompt = f"""
        You are MemoryPalAI's Quiz Agent.

        Create {num_questions} multiple-choice questions (A–D) for the topic "{subject}".
        Use the following roadmap or context as a reference:
        {roadmap}

        Each question should have exactly one correct answer and 3 plausible distractors.
        Format strictly like this:
        Q1. [Question]
        A) Option 1
        B) Option 2
        C) Option 3
        D) Option 4
        ---
        (Continue)
        """

        try:
            result = generate_with_retry(self.llm, prompt, retries=2)
            return str(result).strip()
        except Exception as e:
            print(f"❌ Quiz generation failed: {e}")
            return "Error: Unable to generate quiz."

    def evaluate_answers(self, subject: str, user_answers: dict, quiz_text: str) -> str:
        """
        Evaluates user answers against correct ones inferred from quiz text.
        """
        prompt = f"""
        You are MemoryPalAI's quiz evaluator.

        Subject: {subject}

        Here is the quiz:
        {quiz_text}

        Here are the user's selected answers:
        {user_answers}

        Evaluate:
        1. Which answers are correct or incorrect
        2. Provide total score out of total questions
        3. List topics the user should revise based on mistakes

        Format clearly as:
        - Q1: Correct/Incorrect + brief reason
        - Final Score: x / y
        - Topics to revise: [list]
        """

        try:
            result = generate_with_retry(self.llm, prompt, retries=2)
            return str(result).strip()
        except Exception as e:
            print(f"❌ Quiz evaluation failed: {e}")
            return "Error: Unable to evaluate quiz."
