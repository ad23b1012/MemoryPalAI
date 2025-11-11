# app/services/llm_service.py
import os
import time
from typing import Any
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


# ------------------------------------------------------------
def get_llm(model_name: str = "gemini-2.5-flash"):
    """
    Initialize Gemini (google.generativeai) and return a model handle.
    Raises ValueError if API key is missing.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)

    try:
        model = genai.GenerativeModel(model_name)
        print(f"✅ Gemini LLM initialized ({model_name})")
        return model
    except Exception as e:
        print(f"❌ Failed to initialize Gemini model: {e}")
        raise


# ------------------------------------------------------------
def _extract_text_from_response(response: Any) -> str:
    """
    Safely extract plain text from a Gemini GenerateContentResponse object.
    Ensures the caller always gets a string, never a raw object.
    """
    try:
        # ✅ direct text field (most common case)
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text.strip()

        # ✅ structured parts
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            parts = []
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)
                    elif isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
            return "\n".join(parts).strip()

        # fallback — stringify the whole thing
        return str(response)

    except Exception as e:
        print(f"⚠️ Could not extract Gemini text: {e}")
        return str(response)


# ------------------------------------------------------------
def generate_with_retry(
    llm,
    prompt: str,
    retries: int = 3,
    delay: int = 2,
    generation_config: dict = None,
) -> str:
    """
    Calls Gemini API safely with retry logic.
    Always returns a **string** (never the raw response object).
    """
    generation_config = generation_config or {"temperature": 0.2, "max_output_tokens": 512}

    for attempt in range(1, retries + 1):
        try:
            response = llm.generate_content(prompt, generation_config=generation_config)
            text = _extract_text_from_response(response)
            if not text or text.strip() == "":
                print("⚠️ Gemini returned empty text, falling back to str(response)")
                text = str(response)
            return text.strip()
        except Exception as e:
            print(f"⚠️ Gemini call failed (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
            else:
                return f"❌ LLM error after {retries} retries: {e}"
