# app/database/vector_db.py
"""
VectorDB interface providing compatibility for ChromaDB and unified queries.
"""
from app.database.vector_store import UnifiedVectorStore


class VectorDB(UnifiedVectorStore):
    """
    Wrapper for ChromaDB / local vector store operations.
    """

    def __init__(self, collection_name: str = "memorypal_collection"):
        super().__init__()

    def count(self) -> int:
        stats = self.list_documents()
        return stats.get("total_chunks", 0)
