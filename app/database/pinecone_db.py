# app/database/pinecone_db.py
import os
import sys
import time
import hashlib
from dotenv import load_dotenv
from app.services.embedder import embed_text, EMBEDDING_DIMENSION # Import our function and dimension
from pinecone import Pinecone, ServerlessSpec
# --- IMPORT LANGCHAIN'S SPLITTER AND DOCUMENT ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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
            dimension=EMBEDDING_DIMENSION, # <-- FIX: Use correct 768 dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"❌ Pinecone initialization error: {e}")
    sys.exit(1)

class PineconeDB:
    def __init__(self, namespace=PINECONE_NAMESPACE):
        self.namespace = namespace
        self.index = index
        # --- FIX: Use smart splitter ---
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        print(f"✅ PineconeDB initialized (namespace={namespace}).")

    def _hash_content(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def has_content(self, text: str) -> bool:
        try:
            text_hash = self._hash_content(text)
            results = self.index.query(
                namespace=self.namespace,
                vector=[0.0] * EMBEDDING_DIMENSION,  # <-- FIX: Use correct 768 dimension
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

    def add_document(self, doc_id: str, content: str, metadata: dict = None, topic: str = "general"):
        try:
            if not content.strip():
                print("⚠️ Skipping empty content.")
                return None
            content_hash = self._hash_content(content)
            if self.has_content(content):
                print(f"⚠️ Document already exists, skipping upload: {doc_id}")
                return None

            # --- FIX: Use smart splitter ---
            doc_to_split = Document(page_content=content, metadata=(metadata or {}))
            text_chunks = self.text_splitter.split_documents([doc_to_split])
            
            vectors_to_upsert = []
            for i, chunk in enumerate(text_chunks):
                emb = embed_text(chunk.page_content) 
                if not emb:
                    continue
                vector_id = f"{doc_id}#{i+1}"
                
                chunk_metadata = (metadata or {}).copy()
                chunk_metadata.update({
                    "document_id": doc_id,
                    "chunk_index": i + 1,
                    "chunk_text": chunk.page_content, # Store the clean chunk text
                    "hash": content_hash,
                    "topic": topic,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": emb,
                    "metadata": chunk_metadata
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

    def query(self, query_text: str, top_k: int = 5, topic_filter: str = None):
        try:
            emb = embed_text(query_text)
            if not emb:
                 print(f"❌ Could not generate embedding for query: {query_text}")
                 return {"documents": [], "metadatas": [], "distances": []}
                 
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

    def list_documents(self):
        try:
            stats = self.index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            print(f"📊 Total vectors in index '{PINECONE_INDEX_NAME}': {total}")
            return stats
        except Exception as e:
            print(f"❌ Failed to get index stats: {e}")
            return {}

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

if __name__ == "__main__":
    db = PineconeDB()
    text = "Artificial Intelligence is the simulation of human intelligence by machines."
    db.add_document("demo_doc", text, {"source": "demo.pdf", "tags": ["AI", "intro"]}, topic="AI")
    res = db.query("What is AI?", top_k=3)
    print(res)