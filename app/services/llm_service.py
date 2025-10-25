import os
import httpx
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Gemini API credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiLLM:
    """Simple async client for Google Gemini API."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("❌ Missing GEMINI_API_KEY in environment variables.")
        self.api_key = api_key
        self.model = model
        self.url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )

    async def ask(
        self, prompt: str, temperature: float = 0.3, max_output_tokens: int = 512
    ) -> str:
        """Send a prompt to Gemini and return its text response."""
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # ✅ Gemini API uses API key as query param (not Bearer token)
            response = await client.post(
                f"{self.url}?key={self.api_key}",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # ✅ Safely extract text response
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except (KeyError, IndexError):
                return f"⚠️ Unexpected response format: {data}"


# ----------------- TEST SECTION -----------------
if __name__ == "__main__":
    async def test_llm():
        print("🔍 Testing Gemini LLM Service...\n")
        llm = GeminiLLM(GEMINI_API_KEY, GEMINI_MODEL)
        response = await llm.ask("Explain what an AVL tree is in 3 sentences.")
        print("✅ Response from Gemini:\n")
        print(response)

    asyncio.run(test_llm())
