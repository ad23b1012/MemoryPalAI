# app/services/llm_service.py
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-1")  # Default Grok model
GROK_API_URL = "https://api.together.xyz/v1/engines"

if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY in .env file")


class GrokClient:
    def __init__(self, model: str = GROK_MODEL):
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

    async def ask(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{GROK_API_URL}/{self.model}/completions",
                                         headers=self.headers,
                                         json=payload)
            response.raise_for_status()
            result = response.json()
            # Grok returns text in result['output_text'] or similar
            return result.get("output_text", "").strip()


# Instantiate a global Grok client
grok = GrokClient()


# ----------------- Test -----------------
if __name__ == "__main__":
    import asyncio

    async def test():
        prompt = "Explain the concept of AVL trees in 2 sentences."
        answer = await grok.ask(prompt)
        print("Grok Response:\n", answer)

    asyncio.run(test())
