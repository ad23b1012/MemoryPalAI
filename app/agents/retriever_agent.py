# app/agents/retriever_agent.py

from app.database.pinecone_db import PineconeDB

class RetrieverAgent:
    """
    Retrieves the most relevant text chunks from the Pinecone vector DB.
    """

    def __init__(self):
        self.db = PineconeDB(namespace="memorypal")
        print("✅ RetrieverAgent initialized (Pinecone).")

    def add_document(self, doc_id: str, content: str, metadata: dict = None, topic: str = "general"):
        """Add one document to Pinecone with deduplication."""
        try:
            response = self.db.add_document(doc_id, content, metadata, topic)
            if response:
                print(f"📥 Document '{doc_id}' added to Pinecone index.")
            else:
                print(f"⚠️ Document '{doc_id}' skipped (duplicate or empty).")
        except Exception as e:
            print(f"❌ Error adding document: {e}")

    def query(self, query_text: str, top_k: int = 3, topic_filter: str = None):
        """
        Retrieve relevant content for a given query from Pinecone.
        Returns flattened text results.
        """
        try:
            results = self.db.query(query_text, top_k=top_k, topic_filter=topic_filter)
            docs = results.get("documents", [])
            if not docs:
                print("⚠️ No relevant content found in Pinecone.")
            return results
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
