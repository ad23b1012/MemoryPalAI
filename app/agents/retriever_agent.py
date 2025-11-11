import os
from app.database.vector_db import VectorDB

class RetrieverAgent:
    def __init__(self):
        self.db = VectorDB(collection_name="memorypal_collection")
        print("✅ RetrieverAgent initialized (ChromaDB).")

    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """Add document with metadata — ensuring metadata is non-empty in VectorDB."""
        metadata = metadata or {}
        # don't let metadata be empty dict; VectorDB.add_document will fix, but keep useful keys
        if "source" not in metadata:
            metadata["source"] = doc_id
        try:
            self.db.add_document(doc_id, content, metadata)
            print(f"📥 Added document {doc_id} with metadata {metadata}")
        except Exception as e:
            print(f"❌ Error adding doc {doc_id}: {e}")

    def query(self, query_text: str, top_k: int = 3, source_filter: str = None):
        """Query vector DB and return structured results with provenance."""
        try:
            results = self.db.query(query_text, top_k=top_k)
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []

            # Optional filtering
            if source_filter:
                filtered_docs, filtered_meta, filtered_dist = [], [], []
                for d, m, dist in zip(docs, metadatas, distances):
                    if m.get("source") == source_filter:
                        filtered_docs.append(d)
                        filtered_meta.append(m)
                        filtered_dist.append(dist)
                docs, metadatas, distances = filtered_docs, filtered_meta, filtered_dist

            print(f"🔎 Retrieved {len(docs)} documents.")
            return {"documents": docs, "metadatas": metadatas, "distances": distances}

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
