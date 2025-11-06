import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # General
    PROJECT_NAME: str = "MemoryPalAI"
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 10000))

    # Vector DB
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "memorypalai")
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", 384))

    # LLM Settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")  # can switch to Gemini/Grok

    # Document settings
    MAX_DOCUMENT_SIZE_MB: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", 20))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))

    # Security
    API_KEY_AUTH_ENABLED: bool = os.getenv("API_KEY_AUTH_ENABLED", "False").lower() == "true"
    API_KEYS: list[str] = os.getenv("API_KEYS", "").split(",") if os.getenv("API_KEYS") else []

settings = Settings()
