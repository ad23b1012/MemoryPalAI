# app/services/style_detector.py
import re
import json
import os
from typing import Dict
from app.services.llm_service import get_llm, generate_with_retry

# small heuristic fallback if LLM parsing fails
def _heuristic_style(text: str) -> Dict[str, str]:
    text = (text or "").strip()
    subject = "Unknown"
    style = "Descriptive"
    tone = "Neutral"
    tags = []

    # Very naive heuristics
    # Take first sentence/heading as subject candidate
    first_line = text.splitlines()[0] if text.splitlines() else text[:120]
    if len(first_line) > 5 and len(first_line.split()) < 12:
        subject = re.sub(r'[^A-Za-z0-9\s&\-]', '', first_line).strip()

    # Determine style by content clues
    if "definition" in text.lower() or "is defined" in text.lower():
        style = "Explanatory"
    elif "example" in text.lower() or "for example" in text.lower():
        style = "Example-driven"
    elif "algorithm" in text.lower() or "procedure" in text.lower():
        style = "Technical"

    # tone
    if "please" in text.lower() or "let's" in text.lower():
        tone = "Friendly"
    elif "therefore" in text.lower() or "hence" in text.lower():
        tone = "Formal"

    # tags by frequent nouns - simple word frequency
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    tags = [w for w, c in sorted_words[:6]]

    return {"subject": subject or "Unknown", "style": style, "tone": tone, "tags": tags}


def detect_style_from_text(text: str) -> Dict[str, object]:
    """
    Use LLM to extract subject, style, tone, and tags for a document chunk.
    Returns dict: {subject, style, tone, tags}
    Falls back to heuristics if LLM fails.
    """
    if not text or not text.strip():
        return {"subject": "Unknown", "style": "Unknown", "tone": "Neutral", "tags": []}

    llm = None
    try:
        llm = get_llm("gemini-2.5-flash")
    except Exception:
        llm = None

    prompt = f"""
You are an assistant that extracts a concise JSON summary about a document chunk.
Return ONLY a valid JSON object with keys: subject, style, tone, tags (tags is a list of short keywords).

Analyze the following text and return JSON:

TEXT_START
{text[:2500]}
TEXT_END

Subject should be a short phrase (2-6 words).
Style should be one of: Descriptive, Explanatory, Example-driven, Technical, Narrative, Conversational, Unknown.
Tone should be one of: Neutral, Formal, Friendly, Didactic, Urgent, Unknown.
Tags should be 3-6 short keywords relevant to the text.

Output EXACTLY valid JSON.
"""
    if llm:
        try:
            resp = generate_with_retry(llm, prompt, retries=2)
            if not resp:
                raise RuntimeError("Empty LLM response")
            # try to extract JSON block
            # remove fenced code blocks
            t = re.sub(r"```(?:json)?", "", resp, flags=re.IGNORECASE).strip()
            start = t.find("{")
            end = t.rfind("}")
            if start != -1 and end != -1 and end > start:
                j = t[start:end+1]
            else:
                j = t
            data = json.loads(j)
            # normalize
            subject = data.get("subject", "Unknown")
            style = data.get("style", "Unknown")
            tone = data.get("tone", "Neutral")
            tags = data.get("tags", []) or []
            if not isinstance(tags, list):
                tags = [str(tags)]
            return {"subject": subject, "style": style, "tone": tone, "tags": tags}
        except Exception:
            # fall through to heuristic
            pass

    # heuristic fallback
    return _heuristic_style(text)
