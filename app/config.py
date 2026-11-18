# app/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in current dir or project root
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)

# --- Core App Settings ---
PROJECT_NAME: str = "MemoryPalAI"
VERSION: str = "2.0.0"
DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8501"))

# --- AI & LLM Settings ---
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_API_KEY_FALLBACK: str = os.getenv("GOOGLE_API_KEY_FALLBACK", os.getenv("GOOGLE_API_KEY_2", ""))
# Default to Gemini 3.8 Flash
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

# Additional optional providers
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# --- Vector Database Settings ---
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "memorypalai")
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "memorypal")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
# vector backend: "auto", "pinecone", or "chroma"
VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "auto").lower()
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(root_dir / "chroma_db"))

# --- SQL Database Settings ---
POSTGRES_USER: str = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "")
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

import urllib.parse

# Database URL with automatic URL encoding for passwords and SQLite fallback
raw_db_url = os.getenv("DATABASE_URL")
if raw_db_url:
    if raw_db_url.startswith("postgresql://"):
        DATABASE_URL = raw_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    else:
        DATABASE_URL = raw_db_url
elif POSTGRES_USER and POSTGRES_PASSWORD and POSTGRES_HOST and POSTGRES_DB:
    safe_pass = urllib.parse.quote_plus(POSTGRES_PASSWORD)
    DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{safe_pass}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
else:
    # Safe local fallback so the application never crashes without Postgres
    sqlite_db_path = root_dir / "app" / "database" / "memorypal.db"
    DATABASE_URL = f"sqlite:///{sqlite_db_path}"

# --- Chunking & Document Settings ---
MAX_DOCUMENT_SIZE_MB: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
