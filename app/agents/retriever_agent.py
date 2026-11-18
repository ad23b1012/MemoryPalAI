# app/agents/retriever_agent.py
import logging
from typing import Dict, Any, Optional
from app.database.vector_store import UnifiedVectorStore

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Retrieves the most relevant text chunks from the active vector store
    (Pinecone or persistent ChromaDB) with transparent scoring.
    """

    def __init__(self, namespace: str = "memorypal"):
        self.db = UnifiedVectorStore(namespace=namespace)
        logger.info(f"RetrieverAgent initialized (backend: {self.db.backend})")

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, topic: str = "general") -> Optional[Dict[str, Any]]:
        """Add one document with automatic chunking and deduplication."""
        try:
            return self.db.add_document(doc_id, content, metadata=metadata, topic=topic)
        except Exception as e:
            logger.error(f"Error adding document '{doc_id}': {e}")
            return None

    def query(self, query_text: str, top_k: int = 4, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve relevant content for a given query from vector store.
        """
        try:
            return self.db.query(query_text, top_k=top_k, topic_filter=topic_filter)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "backend": self.db.backend}
