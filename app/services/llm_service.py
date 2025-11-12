# app/services/llm_service.py
import os
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Load API key from environment
def get_llm(model_name="gemini-2.5-flash"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"✅ Gemini LLM initialized ({model_name})")
    return model


def _extract_text_from_response(response):
    """
    Normalize Gemini response: return a single string.
    Supports response.text (simple) or candidates/parts.
    """
    try:
        # quick accessor if available
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        # fallback to candidates/parts
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            lines = []
            for p in parts:
                # some parts may be objects; handle robustly
                if hasattr(p, "text") and p.text:
                    lines.append(p.text)
                else:
                    try:
                        lines.append(str(p))
                    except Exception:
                        pass
            return "\n".join(lines).strip()
        # last resort
        return str(response)
    except Exception:
        return str(response)

def generate_with_retry(model, prompt, retries=3, delay=2):
    """
    Generate text content using Gemini with automatic retry and text extraction.
    """
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            # ✅ Always return clean text
            if hasattr(response, "text"):
                return response.text.strip()
            # Fallback: sometimes response might have candidates[0].content.parts[0].text
            if hasattr(response, "candidates") and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content.parts:
                    return candidate.content.parts[0].text.strip()
            # Final fallback
            return str(response)
        except Exception as e:
            print(f"⚠️ Gemini call failed (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return f"❌ LLM generation failed after {retries} attempts: {e}"
