# app/database/pinecone_db.py

import os
import sys
import time
import hashlib
from dotenv import load_dotenv
from app.services.embedder import embed_text
from pinecone import Pinecone, ServerlessSpec

# -------------------------------------------------
# Pinecone setup
# -------------------------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "memorypal-ai")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "memorypal")

if not PINECONE_API_KEY:
    print("❌ Missing Pinecone API key. Please set PINECONE_API_KEY in your .env")
    sys.exit(1)

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in indexes:
        print(f"🚀 Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"❌ Pinecone initialization error: {e}")
    sys.exit(1)


class PineconeDB:
    """
    MemoryPalAI's Pinecone vector store.
    Handles embedding, upserting, querying, deduplication, and retrieval.
    """

    def __init__(self, namespace=PINECONE_NAMESPACE):
        self.namespace = namespace
        self.index = index
        print(f"✅ PineconeDB initialized (namespace={namespace}).")

    # -------------------------------------------------
    # Helper: Compute hash for deduplication
    # -------------------------------------------------
    def _hash_content(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # -------------------------------------------------
    # ✅ has_content(): Deduplication Check
    # -------------------------------------------------
    def has_content(self, text: str) -> bool:
        """
        Check if a given text (via hash) already exists in the index.
        Prevents re-uploading duplicates.
        """
        try:
            text_hash = self._hash_content(text)
            # query Pinecone with a dummy vector and filter by hash
            results = self.index.query(
                namespace=self.namespace,
                vector=[0.0] * 384,  # dummy vector since we filter by hash
                filter={"hash": {"$eq": text_hash}},
                top_k=1,
                include_metadata=True,
            )
            exists = len(getattr(results, "matches", [])) > 0
            if exists:
                print("⚠️ Duplicate content detected in Pinecone.")
            return exists
        except Exception as e:
            print(f"⚠️ Deduplication check failed: {e}")
            return False

    # -------------------------------------------------
    # Add (upsert) document
    # -------------------------------------------------
    def add_document(self, doc_id: str, content: str, metadata: dict = None, topic: str = "general"):
        """
        Add or upsert a document into Pinecone.
        Performs deduplication using content hash.
        """
        try:
            if not content.strip():
                print("⚠️ Skipping empty content.")
                return None

            # Compute unique hash for deduplication
            content_hash = self._hash_content(content)

            # ✅ Prevent duplicates
            if self.has_content(content):
                print(f"⚠️ Document already exists, skipping upload: {doc_id}")
                return None

            # Split into chunks
            text_chunks = [content[i:i + 1000] for i in range(0, len(content), 1000)]
            vectors_to_upsert = []
            for i, chunk in enumerate(text_chunks):
                emb = embed_text(chunk)
                if not emb:
                    continue
                vector_id = f"{doc_id}#{i+1}"
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": emb,
                    "metadata": {
                        "document_id": doc_id,
                        "chunk_index": i + 1,
                        "chunk_text": chunk,
                        "hash": content_hash,
                        "topic": topic,
                        "tags": metadata.get("tags") if metadata else [],
                        "source": metadata.get("source") if metadata else None,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        **(metadata or {})
                    }
                })

            if not vectors_to_upsert:
                print("⚠️ No valid chunks to upsert.")
                return None

            response = self.index.upsert(namespace=self.namespace, vectors=vectors_to_upsert)
            print(f"📥 Upserted {len(vectors_to_upsert)} chunks for '{doc_id}' successfully.")
            return response

        except Exception as e:
            print(f"❌ Pinecone upsert failed: {e}")
            return None

    # -------------------------------------------------
    # Query documents
    # -------------------------------------------------
    def query(self, query_text: str, top_k: int = 5, topic_filter: str = None):
        try:
            emb = embed_text(query_text)
            fltr = {"topic": {"$eq": topic_filter}} if topic_filter else None
            results = self.index.query(
                namespace=self.namespace,
                vector=emb,
                top_k=top_k,
                include_metadata=True,
                filter=fltr,
            )
            docs, metas, scores = [], [], []
            for match in results.matches:
                docs.append(match.metadata.get("chunk_text", ""))
                metas.append(match.metadata)
                scores.append(match.score)

            print(f"🔍 Retrieved {len(docs)} results for query '{query_text[:60]}...'")
            return {"documents": docs, "metadatas": metas, "distances": scores}
        except Exception as e:
            print(f"❌ Pinecone retrieval error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    # -------------------------------------------------
    # Stats
    # -------------------------------------------------
    def list_documents(self):
        try:
            stats = self.index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            print(f"📊 Total vectors in index '{PINECONE_INDEX_NAME}': {total}")
            return stats
        except Exception as e:
            print(f"❌ Failed to get index stats: {e}")
            return {}

    # -------------------------------------------------
    # Delete
    # -------------------------------------------------
    def delete_document(self, doc_id: str):
        try:
            response = self.index.delete(
                namespace=self.namespace,
                filter={"document_id": {"$eq": doc_id}}
            )
            print(f"🗑️ Deleted document '{doc_id}' successfully.")
            return response
        except Exception as e:
            print(f"❌ Failed to delete document '{doc_id}': {e}")
            return None


# -------------------------------------------------
# Local test
# -------------------------------------------------
if __name__ == "__main__":
    db = PineconeDB()
    text = "Artificial Intelligence is the simulation of human intelligence by machines."
    db.add_document("demo_doc", text, {"source": "demo.pdf", "tags": ["AI", "intro"]}, topic="AI")
    res = db.query("What is AI?", top_k=3)
    print(res)
