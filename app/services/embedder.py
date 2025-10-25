from sentence_transformers import SentenceTransformer
import os

# Use a popular, efficient open-source embedding model
# It will be downloaded automatically the first time you run this
MODEL_NAME = 'all-MiniLM-L6-v2' 

try:
    # Load the model once when the service is imported
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Embedder Service initialized (using local model: {MODEL_NAME}).")
except Exception as e:
    print(f"❌ Error loading Sentence Transformer model: {e}")
    model = None

def get_embedding_model():
    """
    Returns the loaded local Sentence Transformer model.
    """
    if model is None:
        raise RuntimeError("Sentence Transformer model failed to load.")
    return model

def embed_text(text: str) -> list[float]:
    """
    Generates an embedding vector for a single piece of text.
    """
    if model is None:
        raise RuntimeError("Sentence Transformer model failed to load.")

    try:
        # The model's encode function directly returns the vector
        vector = model.encode(text)
        # Convert numpy array to a standard Python list
        return vector.tolist() 
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []

if __name__ == "__main__":
    # Test block to verify the embedding model
    try:
        print("--- Testing Embedder Service ---")
        test_vector = embed_text("This is a test sentence.")
        if test_vector:
             print(f"Successfully generated a vector of dimension: {len(test_vector)}")
        else:
             print("Failed to generate vector.")
    except Exception as e:
        print(f"❌ Error testing Embedder Service: {e}")