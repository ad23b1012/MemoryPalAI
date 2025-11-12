from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = 'all-MiniLM-L6-v2'

try:
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Embedder Service initialized (using local model: {MODEL_NAME}).")
except Exception as e:
    print(f"❌ Error loading Sentence Transformer model: {e}")
    model = None

def get_embedding_model():
    """Return the loaded SentenceTransformer model."""
    if model is None:
        raise RuntimeError("Sentence Transformer model failed to load.")
    return model

def embed_text(text: str) -> list[float]:
    """Generate embedding vector for a single text input."""
    if model is None:
        raise RuntimeError("Sentence Transformer model failed to load.")
    try:
        vector = model.encode(text)
        return vector.tolist()
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []

# ✅ Compatibility alias for Pinecone integration
def get_embeddings(text: str) -> list[float]:
    """Alias for PineconeDB compatibility."""
    return embed_text(text)

if __name__ == "__main__":
    try:
        print("--- Testing Embedder Service ---")
        test_vector = embed_text("This is a test sentence.")
        if test_vector:
            print(f"Successfully generated a vector of dimension: {len(test_vector)}")
        else:
            print("Failed to generate vector.")
    except Exception as e:
        print(f"❌ Error testing Embedder Service: {e}")
