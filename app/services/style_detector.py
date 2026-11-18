# app/services/style_detector.py
import re
import json
import logging
from typing import Dict, Any, List
from app.services.llm_service import get_llm, generate_with_retry

logger = logging.getLogger(__name__)


def _heuristic_style(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    subject = "General Topic"
    style = "Descriptive"
    tone = "Neutral"

    first_line = text.splitlines()[0] if text.splitlines() else text[:120]
    if len(first_line) > 5 and len(first_line.split()) < 12:
        subject = re.sub(r'[^A-Za-z0-9\s&\-]', '', first_line).strip()

    t_lower = text.lower()
    if "definition" in t_lower or "is defined as" in t_lower:
        style = "Explanatory"
    elif "example" in t_lower or "for instance" in t_lower:
        style = "Example-driven"
    elif "algorithm" in t_lower or "code" in t_lower or "architecture" in t_lower:
        style = "Technical"

    if "please" in t_lower or "let's" in t_lower:
        tone = "Friendly"
    elif "therefore" in t_lower or "hence" in t_lower or "furthermore" in t_lower:
        tone = "Formal"

    words = re.findall(r"\b[A-Za-z]{4,}\b", t_lower)
    freq = {}
    for w in words:
        if w not in ("with", "this", "that", "from", "have", "more", "which", "will"):
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    tags = [w.capitalize() for w, _ in sorted_words[:5]]

    return {"subject": subject or "General", "style": style, "tone": tone, "tags": tags}


def detect_style_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts subject, style, tone, and tags for a document chunk using LLM,
    falling back to heuristic extraction on failure.
    """
    if not text or not text.strip():
        return {"subject": "General", "style": "Descriptive", "tone": "Neutral", "tags": []}

    llm = get_llm()
    if not llm:
        return _heuristic_style(text)

    prompt = f"""
Analyze this document excerpt and return a concise JSON object:
{{
  "subject": "Core topic name (2-5 words)",
  "style": "One of: Descriptive, Explanatory, Example-driven, Technical, Narrative, Conversational",
  "tone": "One of: Neutral, Formal, Friendly, Academic",
  "tags": ["tag1", "tag2", "tag3"]
}}

Return ONLY valid JSON.
Excerpt:
{text[:2000]}
"""
    try:
        resp = generate_with_retry(llm, prompt, retries=1)
        clean_resp = re.sub(r"```(?:json)?", "", resp, flags=re.IGNORECASE).strip()
        start = clean_resp.find("{")
        end = clean_resp.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(clean_resp[start:end+1])
            return {
                "subject": data.get("subject", "General"),
                "style": data.get("style", "Descriptive"),
                "tone": data.get("tone", "Neutral"),
                "tags": data.get("tags", []) if isinstance(data.get("tags"), list) else [],
            }
    except Exception as e:
        logger.warning(f"LLM style detection failed ({e}). Falling back to heuristics.")

    return _heuristic_style(text)
