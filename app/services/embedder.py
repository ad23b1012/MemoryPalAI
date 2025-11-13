# app/services/embedder.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY 
import os

MODEL_NAME = "models/text-embedding-004" 
embedder = None
EMBEDDING_DIMENSION = 768 # Gemini's dimension

try:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")
    
    embedder = GoogleGenerativeAIEmbeddings(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
    )
    print(f"✅ Embedder Service initialized (using Google Gemini: {MODEL_NAME}).")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to initialize Gemini embedder: {e}")

def embed_text(text: str) -> list[float]:
    """
    Takes a single string of text and returns its embedding vector.
    """
    if embedder is None:
        print("❌ Embedder not initialized. Returning empty vector.")
        return []
    try:
        return embedder.embed_query(text)
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []