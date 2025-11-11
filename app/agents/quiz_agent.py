from app.services.llm_service import get_llm
import json

class QuizAgent:
    """
    Generates adaptive quizzes and evaluates user's understanding.
    Uses Gemini 2.5 Flash for question generation and answer evaluation.
    """

    def __init__(self):
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ QuizAgent initialized with Gemini-2.5-flash.")

    # ------------------------------------------------------------
    def generate_quiz(self, content: str, topic: str, num_questions: int = 3):
        """Generate quiz questions based on retrieved learning content."""
        print(f"🧩 Generating quiz for topic: {topic} ...")

        prompt = f"""
        You are MemoryPalAI's Quiz Agent.
        Generate {num_questions} multiple-choice questions from the text below.

        The quiz should focus on understanding and reasoning — not just recall.
        Return output in **valid JSON** format only, like this:
        {{
          "quiz": [
            {{
              "question": "What is AI?",
              "options": ["A", "B", "C", "D"],
              "answer": "A"
            }}
          ]
        }}

        Topic: {topic}
        Text:
        {content}
        """

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()
            text = text[text.find("{"): text.rfind("}") + 1]
            data = json.loads(text)
            return data.get("quiz", [])
        except Exception as e:
            print(f"⚠️ Quiz generation failed: {e}")
            return []

    # ------------------------------------------------------------
    def evaluate_answers(self, quiz, user_answers):
        """Compare user answers with correct answers and compute performance."""
        score = 0
        total = len(quiz)
        wrong_topics = []

        for i, q in enumerate(quiz):
            correct = q.get("answer")
            user = user_answers.get(str(i + 1), "").strip()
            if user.lower() == correct.lower():
                score += 1
            else:
                wrong_topics.append(q.get("question", "Unknown"))

        performance = {
            "score": score,
            "total": total,
            "accuracy": round((score / total) * 100, 2) if total > 0 else 0,
            "weak_areas": wrong_topics
        }

        print(f"📊 Quiz Evaluation: {performance}")
        return performance

    # ------------------------------------------------------------
    def recommend_next_steps(self, performance, user_goal, previous_plan):
        """Modify learning plan based on quiz performance."""
        accuracy = performance["accuracy"]
        weak_areas = performance["weak_areas"]

        if accuracy >= 80:
            feedback = f"Excellent work! You’ve mastered most of this topic. Proceed to advanced modules related to {user_goal}."
        elif 50 <= accuracy < 80:
            feedback = f"Good attempt. Review some concepts before moving ahead — especially these topics: {', '.join(weak_areas)}."
        else:
            feedback = f"You need more practice in foundational areas. Revise the following topics: {', '.join(weak_areas)}."

        updated_plan = f"""
        🧠 Adaptive Plan Update:
        - Current Goal: {user_goal}
        - Accuracy: {accuracy}%
        - Weak Areas: {', '.join(weak_areas) if weak_areas else 'None'}

        ✅ Next Steps:
        {feedback}

        📘 Previous Plan Summary:
        {previous_plan[:800]}...
        """

        return updated_plan
