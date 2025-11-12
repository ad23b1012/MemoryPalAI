# app/agents/retriever_agent.py
import os
from app.database.vector_db import VectorDB

class RetrieverAgent:
    def __init__(self):
        self.db = VectorDB(collection_name="memorypal_collection")
        print("✅ RetrieverAgent initialized (ChromaDB).")

    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """Add one document to ChromaDB vector store."""
        try:
            self.db.add_document(doc_id, content, metadata or {})
            print(f"📥 Added document {doc_id}")
        except Exception as e:
            print(f"❌ Error adding document: {e}")

    def query(self, query_text: str, top_k: int = 3, source_filter: str = None):
        """
        Retrieve the most relevant text chunks for a given query.
        Returns flattened results, not nested lists.
        """
        try:
            results = self.db.query(query_text, top_k=top_k)

            # Flatten lists safely (ChromaDB often returns nested)
            docs = []
            if "documents" in results:
                for sub in results["documents"]:
                    if isinstance(sub, list):
                        docs.extend(sub)
                    elif isinstance(sub, str):
                        docs.append(sub)

            metadatas = []
            if "metadatas" in results:
                for sub in results["metadatas"]:
                    if isinstance(sub, list):
                        metadatas.extend(sub)
                    elif isinstance(sub, dict):
                        metadatas.append(sub)

            distances = []
            if "distances" in results:
                for sub in results["distances"]:
                    if isinstance(sub, list):
                        distances.extend(sub)
                    elif isinstance(sub, (int, float)):
                        distances.append(sub)

            # Apply optional source filter
            if source_filter:
                f_docs, f_meta, f_dist = [], [], []
                for d, m, dist in zip(docs, metadatas, distances):
                    if m.get("source") == source_filter:
                        f_docs.append(d)
                        f_meta.append(m)
                        f_dist.append(dist)
                docs, metadatas, distances = f_docs, f_meta, f_dist

            # Simple diagnostic
            print(f"🔎 Retrieved {len(docs)} documents for query: '{query_text[:60]}...'")
            if not docs:
                print("⚠️ No relevant docs found — returning empty list.")

            return {
                "documents": docs,
                "metadatas": metadatas,
                "distances": distances,
            }

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    # Convenience method for direct access
    def retrieve_relevant_docs(self, query_text: str, top_k: int = 3) -> list:
        """Return only text chunks for quick use in LangGraph prompts."""
        res = self.query(query_text, top_k=top_k)
        return res.get("documents", [])
