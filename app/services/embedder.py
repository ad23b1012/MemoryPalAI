# app/services/embedder.py
import os
import asyncio
from google import genai
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Please set your GEMINI_API_KEY in the .env file")

client = genai.Client(api_key=GEMINI_API_KEY)


class Embedder:
    def __init__(self, model: str = "gemini-embedding-001"):
        self.model = model
        # Note: Gemini embedding API returns embeddings of fixed dimension (3072)
        self.dimension = 3072  

    async def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []

        try:
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(None, lambda text=t: client.models.embed_content(
                        model=self.model,
                        contents=text
                    ).embeddings) for t in texts]
            embeddings = await asyncio.gather(*tasks)
            return embeddings
        except Exception as e:
            print(f"Embedding error: {e}")
            return None


# Instantiate a global embedder
embedder = Embedder()


# ----------------- Test -----------------
if __name__ == "__main__":
    async def test():
        texts = [
            "MemoryPalAI is a personal knowledge management system.",
            "It ingests PDFs, notes, audio, and bookmarks."
        ]
        embeddings = await embedder.embed_texts(texts)
        if embeddings:
            print(f"Number of embeddings: {len(embeddings)}")
            print(f"Dimension of first embedding: {len(embeddings[0])}")
            print(embeddings[0][:10], "...")  # print first 10 values
        else:
            print("Failed to generate embeddings.")

    asyncio.run(test())
