import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_llm(model_name="gemini-2.5-flash"):
    """Initialize Gemini safely with retries and proper API configuration."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing GOOGLE_API_KEY environment variable.")

    genai.configure(api_key=api_key)

    try:
        llm = genai.GenerativeModel(model_name)
        print(f"✅ Gemini LLM initialized ({model_name})")
        return llm
    except Exception as e:
        print(f"❌ Failed to initialize Gemini model: {e}")
        raise


def extract_gemini_text(response):
    """Safely extract text from Gemini responses, even if multi-part."""
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            return "\n".join(
                [p.text.strip() for p in parts if hasattr(p, "text") and p.text]
            )
        return str(response)
    except Exception:
        return str(response)


def generate_with_retry(llm, prompt: str, retries: int = 3, delay: int = 3):
    """
    Generate Gemini responses safely with retry logic and proper generation_config.
    Returns clean text output.
    """
    for attempt in range(1, retries + 1):
        try:
            response = llm.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 512,
                }
            )
            return extract_gemini_text(response)
        except Exception as e:
            print(f"⚠️ Gemini call failed (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                raise
