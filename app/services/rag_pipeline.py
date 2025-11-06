# app/services/rag_pipeline.py
from app.agents.ingestion_agent import IngestionAgent
from app.agents.retriever_agent import RetrieverAgent
from app.services.llm_service import get_llm
from app.services.llm_service import generate_with_retry

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline using Gemini-2.5-flash.
    """

    def __init__(self):
        self.ingestion_agent = IngestionAgent()
        self.retriever_agent = RetrieverAgent()
        self.llm = get_llm("gemini-2.5-flash")
        print("✅ RAGPipeline initialized with Gemini-2.5-flash.")

    def ingest_and_store(self, file_path: str):
        """
        Ingests a document and stores its embeddings in ChromaDB.
        """
        docs = self.ingestion_agent.ingest(file_path)
        if not docs:
            return "❌ Failed to ingest document."

        for i, doc in enumerate(docs):
            doc_id = f"{file_path}_{i}"
            self.retriever_agent.add_document(doc_id, doc.page_content, doc.metadata)
        return f"✅ Ingested and stored {len(docs)} document(s)."

    def ask(self, query: str, top_k: int = 3):
        """
        Retrieves context and queries Gemini for an answer.
        """
        results = self.retriever_agent.query(query, top_k)
        if not results or not results.get("documents"):
            return "❌ No relevant information found."

        # Combine top context documents
        context = "\n\n".join(results["documents"][0])

        prompt = f"""
        You are MemoryPalAI, an intelligent knowledge assistant.
        Answer the following question using the provided context.

        Context:
        {context}

        Question:
        {query}

        Provide a concise, factual, and clear answer.
        """

        try:
            response = generate_with_retry(self.llm, prompt)
            return response.text.strip() if hasattr(response, "text") else "⚠️ No answer generated."
        except Exception as e:
            return f"❌ Error generating answer: {e}"


# ---------------- Test ----------------
if __name__ == "__main__":
    print("🚀 Testing RAGPipeline with Gemini-2.5-flash...\n")
    rag = RAGPipeline()

    # Test ingestion
    test_file = "test_note.txt"
    with open(test_file, "w") as f:
        f.write("Artificial Intelligence is the simulation of human intelligence processes by machines.")
    print(rag.ingest_and_store(test_file))

    # Test query
    answer = rag.ask("What is Artificial Intelligence?")
    print("\n💬 Gemini Answer:\n", answer)
