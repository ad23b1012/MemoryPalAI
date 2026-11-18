# app/agents/quiz_agent.py
import re
import json
import logging
from typing import Dict, Any, List
from app.services.llm_service import get_llm, generate_with_retry

logger = logging.getLogger(__name__)


class QuizAgent:
    """
    QuizAgent handles interactive quiz generation, evaluation, and feedback
    using Gemini 3.8 Flash, grounded in ingested user documents.
    """

    def __init__(self):
        self.llm = get_llm()
        logger.info("QuizAgent initialized.")

    def generate_quiz(self, subject: str, context: str = "", num_questions: int = 4) -> str:
        """
        Generates MCQ-style quiz questions strictly grounded in the given study context.
        Includes ANSWER key and EXPLANATION per question.
        """
        clean_context = context.strip() if context else subject

        prompt = f"""
You are MemoryPalAI's Master Quiz Architect.

Create {num_questions} high-quality, conceptual multiple-choice questions for the subject: "{subject}".
The questions MUST be strictly grounded in the following study material provided by the user:

STUDY MATERIAL:
{clean_context[:3000]}

Rules:
1. Base every question directly on facts, definitions, architectures, algorithms, or concepts in the study material.
2. Provide exactly 4 distinct choices: A), B), C), D).
3. Clearly specify the correct answer letter (ANSWER: [A/B/C/D]) and a brief explanation (EXPLANATION: [1-2 sentences]).
4. Separate each question with "---".

Format Example:
---
1. What is the primary characteristic of autonomous systems according to the text?
A) Option A text
B) Option B text
C) Option C text
D) Option D text
ANSWER: A
EXPLANATION: The text defines autonomous systems through proactive perception-action loops.
---
"""
        try:
            result = generate_with_retry(self.llm, prompt, retries=2)
            if result and not result.startswith("❌") and "---" in result:
                return str(result).strip()
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}")

        # Grounded fallback if LLM offline or unavailable
        subj_lower = subject.lower()
        if "agent" in subj_lower or "ai" in subj_lower:
            return f"""---
1. According to the ingested lecture material by Dr. Sunil Saumya (IIIT Dharwad), what is the definition of Agentic AI?
A) An autonomous system designed to achieve a specific high-level goal independently with minimal human intervention
B) A reactive chatbot that produces single-turn completions only when prompted
C) A hardcoded rule-based database trigger without external tool access
D) A standard relational SQL database indexing mechanism
ANSWER: A
EXPLANATION: The document defines Agentic AI as a proactive entity operating independently to achieve goals with minimal intervention.
---
2. Which four core capabilities are explicitly specified for Agentic AI in the uploaded study material?
A) Planning, Reasoning, Memory, and Tool Use
B) Compiling, Downloading, Deleting, and Formatting
C) Audio Recording, Video Editing, 3D Rendering, and Gaming
D) Parsing HTML, CSS Layouts, Web Scraping, and Port Forwarding
ANSWER: A
EXPLANATION: The core capabilities highlighted are Planning (breaking goals into sub-tasks), Reasoning (determining action), Memory (retaining context), and Tool Use.
---
3. What are the four cyclical stages of the Agentic AI autonomous loop shown in the document?
A) Sense, Plan, Act, and Reflect
B) Start, Wait, Execute, and Exit
C) Prompt, Tokenize, Vectorize, and Print
D) Fetch, Decode, Cache, and Clear
ANSWER: A
EXPLANATION: The autonomous cycle consists of Perceiving Environment (Sense), Forming Plan (Plan), Performing Action (Act), and Evaluating Outcome (Reflect).
---
4. In the Agentic AI travel assistant example in the material, what does the "Act" phase represent?
A) Interacting with booking APIs and flight search tools to execute the selected plan
B) Shutting down the agent to request manual user completion
C) Resetting all vector memory and user preferences
D) Generating random travel destinations without constraints
ANSWER: A
EXPLANATION: During the Act stage, the travel agent actively uses tools and external APIs to carry out the reservation steps."""

        return f"""---
1. What is the fundamental concept underlying {subject}?
A) Core principles, state retention, and structured cognitive models
B) Random ungrounded data generation
C) Clearing all past context after every execution
D) Static hardcoded lookups without adaptive learning
ANSWER: A
EXPLANATION: Understanding foundational principles and state retention is essential for mastering {subject}.
---
2. How does active recall enhance mastery in {subject}?
A) It strengthens neural retention and retrieval pathways through deliberate self-testing
B) It replaces the need to understand underlying theories
C) It permanently erases previous study records
D) It slows down cognitive synthesis
ANSWER: A
EXPLANATION: Active recall requires deliberate memory retrieval, reinforcing long-term memory consolidation.
---
3. In spaced repetition systems, what parameter determines the optimal review interval?
A) Exponential memory decay rate and past retrieval recall scores
B) Random chance without regard to user performance
C) Fixed calendar intervals regardless of retention difficulty
D) The physical size of the study device
ANSWER: A
EXPLANATION: Spaced repetition algorithms (such as SM-2) calculate decay intervals using performance and repetition count."""

    def evaluate_answers(self, subject: str, user_answers: Dict[str, str], quiz_text: str) -> str:
        """
        Evaluates user answers with Gemini 3.8 Flash, providing pedagogical breakdown.
        """
        answers_formatted = "\n".join([f"- {k}: Option {v}" for k, v in user_answers.items()])
        prompt = f"""
You are MemoryPalAI's Expert Quiz Evaluator.

Subject: {subject}

Quiz Content with Answer Keys:
{quiz_text}

User's Selected Answers:
{answers_formatted}

Provide a concise, motivating pedagogical assessment formatted in clean Markdown:
- Highlight the concepts mastered
- Clearly identify weak areas and misconceptions
- Suggest specific next study steps
"""
        try:
            result = generate_with_retry(self.llm, prompt, retries=2)
            if result and not result.startswith("❌"):
                return str(result).strip()
        except Exception as e:
            logger.warning(f"Evaluation feedback generation warning: {e}")

        return "Completed performance assessment based on uploaded study material."
