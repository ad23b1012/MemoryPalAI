# app/services/embedder.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GOOGLE_API_KEY 
import os

# --- Model Configuration ---
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

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing Embedder Service ---")
    if embedder:
        test_vector = embed_text("This is a test sentence.")
        if test_vector:
             print(f"Successfully generated a vector of dimension: {len(test_vector)}")
        else:
             print("Failed to generate test vector.")
    else:
        print("Embedder could not be loaded. Check API key and configuration.")