# app/database/vector_store.py
import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    VECTOR_BACKEND,
    CHROMA_PERSIST_DIR,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.services.embedder import embed_text, embed_documents
from app.database.document_registry import (
    register_document,
    unregister_document,
    get_all_documents,
    get_all_topics,
    load_registry
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class UnifiedVectorStore:
    """
    Production-grade Unified Vector Store supporting Pinecone (cloud)
    and ChromaDB (persistent local) with automatic fallback and zero-crash initialization.
    """

    def __init__(self, namespace: str = PINECONE_NAMESPACE):
        self.namespace = namespace
        self.backend = "unknown"
        self.index = None
        self.chroma_collection = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self._init_backend()

    def _init_backend(self):
        """Attempts to initialize Pinecone if requested and configured; falls back to ChromaDB."""
        prefer_pinecone = (
            VECTOR_BACKEND in ("auto", "pinecone")
            and bool(PINECONE_API_KEY)
            and PINECONE_API_KEY != "your_pinecone_api_key_here"
        )

        if prefer_pinecone:
            try:
                from pinecone import Pinecone, ServerlessSpec
                pc = Pinecone(api_key=PINECONE_API_KEY)
                
                # Check indexes safely
                raw_indexes = pc.list_indexes()
                if hasattr(raw_indexes, "names"):
                    existing_names = raw_indexes.names()
                else:
                    existing_names = [getattr(i, "name", str(i)) for i in raw_indexes]

                if PINECONE_INDEX_NAME not in existing_names:
                    logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIMENSION})...")
                    pc.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                    )
                self.index = pc.Index(PINECONE_INDEX_NAME)
                self.backend = "pinecone"
                logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME} (namespace={self.namespace})")
                return
            except Exception as e:
                logger.warning(f"Pinecone connection failed ({e}). Gracefully falling back to persistent ChromaDB.")

        # Local ChromaDB Fallback
        try:
            import chromadb
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self.chroma_collection = chroma_client.get_or_create_collection(
                name="memorypal_collection",
                metadata={"hnsw:space": "cosine"}
            )
            self.backend = "chroma"
            logger.info(f"ChromaDB persistent vector store initialized at {CHROMA_PERSIST_DIR}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB fallback: {e}")
            self.backend = "memory_fallback"
            self._memory_docs = []

    def _hash_content(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def has_content(self, text: str) -> bool:
        """Check if content has already been indexed."""
        if not text:
            return False
        content_hash = self._hash_content(text)

        if self.backend == "pinecone" and self.index:
            try:
                results = self.index.query(
                    namespace=self.namespace,
                    vector=[0.0] * EMBEDDING_DIMENSION,
                    filter={"hash": {"$eq": content_hash}},
                    top_k=1,
                    include_metadata=True,
                )
                matches = getattr(results, "matches", []) or []
                return len(matches) > 0
            except Exception as e:
                logger.warning(f"Pinecone deduplication check warning: {e}")
                return False

        elif self.backend == "chroma" and self.chroma_collection:
            try:
                res = self.chroma_collection.get(where={"hash": content_hash}, limit=1)
                return bool(res and res.get("ids"))
            except Exception:
                return False

        return False

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, topic: str = "general") -> Optional[Dict[str, Any]]:
        """Splits content into chunks, generates embeddings, and indexes into active backend."""
        if not content or not content.strip():
            logger.warning(f"Empty content for doc_id '{doc_id}', skipping.")
            return None

        content_hash = self._hash_content(content)
        if self.has_content(content):
            logger.info(f"Duplicate document detected ({doc_id}), skipping index.")
            return {"status": "duplicate", "doc_id": doc_id, "chunks": 0}

        doc = Document(page_content=content, metadata=(metadata or {}))
        text_chunks = self.text_splitter.split_documents([doc])
        if not text_chunks:
            return None

        clean_topic = topic or (metadata.get("subject") if metadata else None) or "general"
        base_meta = metadata.copy() if metadata else {}
        base_meta.update({
            "document_id": doc_id,
            "hash": content_hash,
            "topic": clean_topic,
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Batch-generate embeddings for all chunks for speed and low latency
        chunk_texts = [c.page_content for c in text_chunks]
        all_embeddings = embed_documents(chunk_texts)
        if len(all_embeddings) != len(text_chunks):
            all_embeddings = [embed_text(c.page_content) for c in text_chunks]

        # Process and upsert vectors
        if self.backend == "pinecone" and self.index:
            vectors_to_upsert = []
            for i, chunk in enumerate(text_chunks):
                emb = all_embeddings[i] if i < len(all_embeddings) else embed_text(chunk.page_content)
                if not emb:
                    continue
                chunk_meta = base_meta.copy()
                chunk_meta.update({
                    "chunk_index": i + 1,
                    "chunk_text": chunk.page_content,
                    # Ensure metadata values are serializable primitives
                    "tags": ",".join(chunk_meta.get("tags", [])) if isinstance(chunk_meta.get("tags"), list) else str(chunk_meta.get("tags", "")),
                })
                vectors_to_upsert.append({
                    "id": f"{doc_id}#chunk_{i+1}",
                    "values": emb,
                    "metadata": chunk_meta
                })

            if vectors_to_upsert:
                # Upsert in batches of 50
                batch_size = 50
                for start in range(0, len(vectors_to_upsert), batch_size):
                    self.index.upsert(
                        vectors=vectors_to_upsert[start:start + batch_size],
                        namespace=self.namespace
                    )
                # Register document in persistent registry
                register_document(
                    doc_id=doc_id,
                    topic=clean_topic,
                    chunks=len(vectors_to_upsert),
                    metadata=base_meta,
                    snippet=content[:1000]
                )
            return {"status": "success", "doc_id": doc_id, "chunks": len(vectors_to_upsert), "backend": "pinecone"}

        elif self.backend == "chroma" and self.chroma_collection:
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            for i, chunk in enumerate(text_chunks):
                emb = all_embeddings[i] if i < len(all_embeddings) else embed_text(chunk.page_content)
                chunk_meta = base_meta.copy()
                chunk_meta.update({
                    "chunk_index": i + 1,
                    "chunk_text": chunk.page_content,
                    "tags": ",".join(chunk_meta.get("tags", [])) if isinstance(chunk_meta.get("tags"), list) else str(chunk_meta.get("tags", "")),
                })
                ids.append(f"{doc_id}#chunk_{i+1}")
                documents.append(chunk.page_content)
                metadatas.append(chunk_meta)
                embeddings.append(emb)

            if ids:
                self.chroma_collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                # Register document in persistent registry
                register_document(
                    doc_id=doc_id,
                    topic=clean_topic,
                    chunks=len(ids),
                    metadata=base_meta,
                    snippet=content[:1000]
                )
            return {"status": "success", "doc_id": doc_id, "chunks": len(ids), "backend": "chroma"}

        return {"status": "failed", "doc_id": doc_id, "backend": self.backend}

    def query(self, query_text: str, top_k: int = 4, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """Queries for similar text chunks, optionally filtering by topic."""
        if not query_text or not query_text.strip():
            return {"documents": [], "metadatas": [], "distances": []}

        query_emb = embed_text(query_text)

        if self.backend == "pinecone" and self.index:
            try:
                filter_dict = {}
                if topic_filter and topic_filter.strip():
                    filter_dict["topic"] = {"$eq": topic_filter.strip()}

                res = self.index.query(
                    namespace=self.namespace,
                    vector=query_emb,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None,
                )
                matches = getattr(res, "matches", []) or []
                docs = []
                metas = []
                distances = []
                for m in matches:
                    meta = getattr(m, "metadata", {}) or {}
                    text = meta.get("chunk_text") or meta.get("text", "")
                    docs.append(text)
                    metas.append(meta)
                    distances.append(getattr(m, "score", 0.0))
                return {"documents": docs, "metadatas": metas, "distances": distances, "backend": "pinecone"}
            except Exception as e:
                logger.error(f"Pinecone query error: {e}")

        elif self.backend == "chroma" and self.chroma_collection:
            try:
                where_filter = None
                if topic_filter and topic_filter.strip():
                    where_filter = {"topic": topic_filter.strip()}

                res = self.chroma_collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    where=where_filter
                )
                docs = (res.get("documents", [[]]) or [[]])[0]
                metas = (res.get("metadatas", [[]]) or [[]])[0]
                distances = (res.get("distances", [[]]) or [[]])[0]
                return {"documents": docs, "metadatas": metas, "distances": distances, "backend": "chroma"}
            except Exception as e:
                logger.error(f"ChromaDB query error: {e}")

        return {"documents": [], "metadatas": [], "distances": [], "backend": self.backend}

    def list_documents(self) -> Dict[str, Any]:
        """Returns clean overview statistics and list of indexed documents."""
        reg_docs = get_all_documents()
        reg_topics = get_all_topics()
        total_chunks = 0

        if self.backend == "pinecone" and self.index:
            try:
                stats = self.index.describe_index_stats()
                ns_stats = getattr(stats, "namespaces", {}).get(self.namespace, {})
                count = getattr(ns_stats, "vector_count", getattr(stats, "total_vector_count", 0))
                total_chunks = count
                return {
                    "backend": "Pinecone (Cloud)",
                    "namespace": self.namespace,
                    "total_chunks": total_chunks,
                    "total_documents": len(reg_docs),
                    "documents": reg_docs,
                    "topics": reg_topics,
                    "index_name": PINECONE_INDEX_NAME,
                    "status": "Healthy & Online",
                }
            except Exception as e:
                return {
                    "backend": "Pinecone",
                    "status": f"Error: {e}",
                    "total_chunks": 0,
                    "total_documents": len(reg_docs),
                    "documents": reg_docs,
                    "topics": reg_topics,
                }

        elif self.backend == "chroma" and self.chroma_collection:
            try:
                all_data = self.chroma_collection.get()
                total_chunks = len(all_data.get("ids", []))
                metas = all_data.get("metadatas", []) or []
                chroma_docs = {}
                chroma_topics = set(reg_topics)
                for m in metas:
                    doc_id = m.get("document_id", "Unknown")
                    topic = m.get("topic", "General")
                    chroma_topics.add(topic)
                    if doc_id not in chroma_docs:
                        chroma_docs[doc_id] = {
                            "document_id": doc_id,
                            "topic": topic,
                            "chunks": 0,
                            "indexed_at": m.get("indexed_at", "N/A"),
                            "style": m.get("style", "N/A"),
                            "tone": m.get("tone", "N/A"),
                        }
                    chroma_docs[doc_id]["chunks"] += 1

                # Merge registry and chroma docs
                merged_docs = {d["document_id"]: d for d in reg_docs}
                for k, v in chroma_docs.items():
                    if k not in merged_docs:
                        merged_docs[k] = v

                return {
                    "backend": "ChromaDB (Persistent Local)",
                    "total_documents": len(merged_docs),
                    "total_chunks": total_chunks,
                    "documents": list(merged_docs.values()),
                    "topics": sorted(list(chroma_topics)),
                    "status": "Healthy & Online",
                }
            except Exception as e:
                return {
                    "backend": "ChromaDB",
                    "status": f"Error: {e}",
                    "total_chunks": 0,
                    "total_documents": len(reg_docs),
                    "documents": reg_docs,
                    "topics": reg_topics,
                }

        return {
            "backend": "None",
            "total_chunks": 0,
            "total_documents": len(reg_docs),
            "documents": reg_docs,
            "topics": reg_topics,
            "status": "Not configured"
        }

    def delete_document(self, doc_id: str) -> bool:
        """Deletes all chunks associated with a document_id."""
        if not doc_id:
            return False

        # Unregister from document registry
        unregister_document(doc_id)

        if self.backend == "pinecone" and self.index:
            try:
                # Query chunks by document_id prefix or filter
                self.index.delete(filter={"document_id": {"$eq": doc_id}}, namespace=self.namespace)
                return True
            except Exception as e:
                logger.error(f"Pinecone delete error: {e}")
                return False

        elif self.backend == "chroma" and self.chroma_collection:
            try:
                self.chroma_collection.delete(where={"document_id": doc_id})
                return True
            except Exception as e:
                logger.error(f"ChromaDB delete error: {e}")
                return False

        return False

    def clear(self) -> bool:
        """Wipes the entire vector store."""
        if self.backend == "pinecone" and self.index:
            try:
                self.index.delete(delete_all=True, namespace=self.namespace)
                return True
            except Exception as e:
                logger.error(f"Failed to clear Pinecone: {e}")
                return False
        elif self.backend == "chroma" and self.chroma_collection:
            try:
                self.chroma_collection.delete(where={})
                return True
            except Exception as e:
                logger.error(f"Failed to clear ChromaDB: {e}")
                return False
        return False
