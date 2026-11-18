# app/database/document_registry.py
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "documents_registry.json")


def load_registry() -> Dict[str, Any]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading documents_registry.json: {e}")
        return {}


def save_registry(registry: Dict[str, Any]) -> bool:
    try:
        os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving documents_registry.json: {e}")
        return False


def register_document(doc_id: str, topic: str, chunks: int, metadata: Optional[Dict[str, Any]] = None, snippet: str = ""):
    """Registers an ingested document with its metadata, topic, and snippet."""
    reg = load_registry()
    meta = metadata or {}
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    reg[doc_id] = {
        "document_id": doc_id,
        "topic": topic or "General Knowledge",
        "chunks": chunks or 1,
        "indexed_at": meta.get("indexed_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "style": meta.get("style", "Descriptive"),
        "tone": meta.get("tone", "Neutral"),
        "tags": tags,
        "snippet": snippet[:1000] if snippet else ""
    }
    save_registry(reg)

    # Also sync to PostgreSQL if active
    try:
        from app.database.postgres_client import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id VARCHAR(255) PRIMARY KEY,
                    topic VARCHAR(255),
                    chunks INT,
                    style VARCHAR(100),
                    tone VARCHAR(100),
                    indexed_at VARCHAR(100),
                    snippet TEXT
                );
            """))
            conn.execute(
                text("""
                    INSERT INTO documents (document_id, topic, chunks, style, tone, indexed_at, snippet)
                    VALUES (:doc_id, :topic, :chunks, :style, :tone, :indexed_at, :snippet)
                    ON CONFLICT (document_id) DO UPDATE SET
                        topic = EXCLUDED.topic,
                        chunks = EXCLUDED.chunks,
                        style = EXCLUDED.style,
                        tone = EXCLUDED.tone,
                        indexed_at = EXCLUDED.indexed_at,
                        snippet = EXCLUDED.snippet;
                """),
                {
                    "doc_id": doc_id,
                    "topic": topic or "General Knowledge",
                    "chunks": chunks or 1,
                    "style": meta.get("style", "Descriptive"),
                    "tone": meta.get("tone", "Neutral"),
                    "indexed_at": meta.get("indexed_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
                    "snippet": snippet[:1000] if snippet else ""
                }
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"Postgres document sync skipped (using JSON registry): {e}")


def unregister_document(doc_id: str):
    """Removes a document from the registry."""
    reg = load_registry()
    if doc_id in reg:
        del reg[doc_id]
        save_registry(reg)

    try:
        from app.database.postgres_client import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM documents WHERE document_id = :doc_id"), {"doc_id": doc_id})
            conn.commit()
    except Exception:
        pass


def get_all_documents() -> List[Dict[str, Any]]:
    reg = load_registry()
    return list(reg.values())


def get_all_topics() -> List[str]:
    reg = load_registry()
    topics = set()
    for doc in reg.values():
        t = doc.get("topic")
        if t and t.strip():
            topics.add(t.strip())
    return sorted(list(topics))


def get_document_snippet(doc_id: Optional[str] = None) -> str:
    """Returns the text snippet of a document or the most recent document."""
    reg = load_registry()
    if not reg:
        return ""
    if doc_id and doc_id in reg:
        return reg[doc_id].get("snippet", "")
    # Default to first available document snippet
    first_doc = next(iter(reg.values()))
    return first_doc.get("snippet", "")
