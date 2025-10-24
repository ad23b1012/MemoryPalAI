import asyncio
from typing import List
from app.services.parser import DocumentParser
from app.utils.chunker import chunk_text
from app.services.embedder import embedder
from app.database.pinecone_client import index

class IngestionAgent:
    def __init__(self):
        self.parser = DocumentParser()

    async def ingest_document(self, doc_url: str) -> List[str]:
        """Parse document, chunk text, embed, and store in Pinecone."""
        text, content_type = await self.parser.parse_document(doc_url)
        if not text:
            raise ValueError(f"Failed to parse document: {doc_url}")

        # Chunk text
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No text chunks generated.")

        # Generate embeddings
        embeddings = await embedder.embed_texts(chunks)
        if not embeddings:
            raise ValueError("Failed to generate embeddings.")

        # Store embeddings in Pinecone
        for i, chunk in enumerate(chunks):
            metadata = {"source": doc_url, "chunk_id": i}
            index.upsert([(f"{doc_url}_{i}", embeddings[i], metadata)])

        return chunks

# ---------------------------
# Inline test code
# ---------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    agent = IngestionAgent()

    test_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    async def test():
        chunks = await agent.ingest_document(test_url)
        print(f"✅ Ingested {len(chunks)} chunks from {test_url}")

    asyncio.run(test())
