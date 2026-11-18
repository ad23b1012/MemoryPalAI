# app/services/rag_pipeline.py
import logging
from typing import Dict, Any
from app.agents.ingestion_agent import IngestionAgent
from app.agents.retriever_agent import RetrieverAgent
from app.services.llm_service import get_llm, generate_with_retry

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline grounded on indexed knowledge.
    """

    def __init__(self):
        self.ingestion_agent = IngestionAgent()
        self.retriever_agent = RetrieverAgent()
        self.llm = get_llm()
        logger.info("RAGPipeline initialized.")

    def ingest_and_store(self, file_path: str, topic: str = "general") -> str:
        """
        Ingests a document and indexes into the active vector store.
        """
        docs = self.ingestion_agent.ingest(file_path)
        if not docs:
            return "❌ Failed to ingest document."

        count = 0
        for i, doc in enumerate(docs):
            doc_id = f"{doc.metadata.get('source', 'doc')}_{i}"
            res = self.retriever_agent.add_document(doc_id, doc.page_content, doc.metadata, topic=topic)
            if res:
                count += 1
        return f"✅ Successfully ingested and indexed {count} chunk(s)."

    def ask(self, query: str, top_k: int = 4, topic_filter: str = None) -> Dict[str, Any]:
        """
        Retrieves context and queries Gemini for grounded answering.
        """
        results = self.retriever_agent.query(query, top_k=top_k, topic_filter=topic_filter)
        docs = results.get("documents", [])

        if not docs:
            # General fallback
            fallback_prompt = f"You are a helpful learning assistant. Provide a helpful, accurate answer to: {query}"
            answer = generate_with_retry(self.llm, fallback_prompt, retries=2)
            return {
                "answer": answer,
                "grounded": False,
                "context": "",
                "sources": []
            }

        context = "\n\n---\n\n".join(docs[:4])
        grounded_prompt = f"""
You are MemoryPalAI — an expert retrieval-grounded AI tutor.
Use the following retrieved STUDY CONTEXT to answer the QUESTION accurately.
If the information cannot be found in the context, explicitly state what is missing and then supplement with standard foundational knowledge.

STUDY CONTEXT:
{context}

QUESTION:
{query}
"""
        answer = generate_with_retry(self.llm, grounded_prompt, retries=2)
        return {
            "answer": answer,
            "grounded": True,
            "context": context,
            "sources": results.get("metadatas", [])
        }
