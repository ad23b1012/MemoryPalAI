# app/database/pinecone_db.py
"""
PineconeDB interface providing backward compatibility with the legacy module,
powered by the unified vector store with zero-crash fallback.
"""
import logging
from app.database.vector_store import UnifiedVectorStore
from app.config import PINECONE_NAMESPACE

logger = logging.getLogger(__name__)


class PineconeDB(UnifiedVectorStore):
    """
    Backward-compatible wrapper for PineconeDB.
    Inherits all capabilities from UnifiedVectorStore.
    """

    def __init__(self, namespace: str = PINECONE_NAMESPACE):
        super().__init__(namespace=namespace)
        logger.info(f"PineconeDB initialized (active backend: {self.backend})")
