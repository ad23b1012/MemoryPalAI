# Import necessary services and utilities
from app.services.llm_service import get_llm
from app.services.embedder import get_embedding_model # Imports the SentenceTransformer model
from app.database.vector_db import get_vector_collection
from app.utils.chunker import chunk_documents

# Import LangChain and related components
from langchain import hub
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings # Base class for our adapter
from langchain_core.runnables import RunnablePassthrough # For manual chain construction
from langchain_community.vectorstores import Chroma # Import Chroma vector store
from sentence_transformers import SentenceTransformer # Import the model type
from operator import itemgetter # For manual chain construction
import os # For test file handling

# --- Attempt to import the chain creation functions ---
# These paths are based on typical v1.x structures, but might vary slightly
# If these fail at runtime, we may need further adjustment
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    # create_retrieval_chain is often composed, so we might not need a direct import
except ImportError:
    print("Warning: Could not import chain functions from expected paths.")
    # Define placeholder functions or raise a more specific error if needed later
    def create_stuff_documents_chain(*args, **kwargs):
        raise NotImplementedError("create_stuff_documents_chain import failed.")

# Helper class to wrap the SentenceTransformer model for LangChain/ChromaDB
class LangchainEmbeddingAdapter(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

class RetrieverAgent:
    """
    An agent that indexes knowledge for search and answers questions
    using Retrieval-Augmented Generation (RAG) with a local embedding model.
    """
    def __init__(self):
        self.llm = get_llm()
        self.embedding_model_instance = get_embedding_model()
        self.embeddings = LangchainEmbeddingAdapter(self.embedding_model_instance)
        self.collection = get_vector_collection()
        if self.collection is None:
             raise ConnectionError("Failed to connect to VectorDB collection.")
        print("✅ RetrieverAgent initialized with local embedder.")

    def index_knowledge(self, documents: list[Document]):
        """Splits, embeds, and indexes documents in the VectorDB."""
        if not documents:
            print("⚠️ Retriever Agent (Indexing): No documents provided.")
            return

        print("\n🚀 Retriever Agent (Indexing): Creating searchable index...")
        split_docs = chunk_documents(documents)
        doc_contents = [doc.page_content for doc in split_docs]
        doc_metadatas = [doc.metadata for doc in split_docs]
        current_count = self.collection.count()
        doc_ids = [f"doc_{current_count + i}" for i in range(len(split_docs))]

        try:
            self.collection.add(
                 ids=doc_ids,
                 documents=doc_contents,
                 metadatas=doc_metadatas
            )
            print(f"✅ Retriever Agent (Indexing): Added {len(doc_ids)} chunks to VectorDB.")
        except Exception as e:
             print(f"❌ Retriever Agent (Indexing): Error adding to VectorDB: {e}")
             print("Attempting update for existing IDs (upsert)...")
             try:
                 self.collection.upsert(
                      ids=doc_ids,
                      documents=doc_contents,
                      metadatas=doc_metadatas
                 )
                 print(f"✅ Retriever Agent (Indexing): Upserted {len(doc_ids)} chunks in VectorDB.")
             except Exception as ue:
                 print(f"❌ Retriever Agent (Indexing): Upsert failed: {ue}")


    def answer_question(self, question: str):
        """Answers a question based on the indexed knowledge."""
        print(f"\n🚀 Retriever Agent (Q&A): Answering question: '{question}'")

        vectorstore = Chroma(
             client=self.collection.client,
             collection_name=self.collection.name,
             embedding_function=self.embeddings
        )
        retriever = vectorstore.as_retriever()
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        # --- Use create_stuff_documents_chain if imported successfully ---
        try:
             combine_docs_chain = create_stuff_documents_chain(
                 llm=self.llm,
                 prompt=retrieval_qa_chat_prompt
             )
        except NameError:
             print("❌ ERROR: create_stuff_documents_chain was not imported correctly.")
             return "Error: System configuration issue."
        except Exception as e:
            print(f"❌ ERROR: Failed to create combine_docs_chain: {e}")
            return "Error: System configuration issue."


        # Manually create the retrieval chain logic using Runnables
        retrieval_chain = RunnablePassthrough.assign(
            context=itemgetter("input") | retriever,
        ) | combine_docs_chain

        try:
             response = retrieval_chain.invoke({"input": question})
             print("✅ Retriever Agent (Q&A): Successfully generated an answer.")
             return response["answer"]
        except Exception as e:
             print(f"❌ Retriever Agent (Q&A): Error generating answer: {e}")
             # You might want to log the full error `e` for debugging
             return f"Error: Could not generate answer."

if __name__ == "__main__":
    print("--- Testing RetrieverAgent ---")
    try:
        from app.agents.ingestion_agent import IngestionAgent
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            test_pdf_path = "test_report.pdf"
            c = canvas.Canvas(test_pdf_path, pagesize=letter)
            c.drawString(100, 750, "MemoryPalAI Project Report")
            c.drawString(100, 735, "Author: Dr. Sunil Saumya")
            c.drawString(100, 720, "This system uses a VectorDB for efficient search.")
            c.save()
            print(f"Created dummy PDF: {test_pdf_path}")

            ingestion_agent = IngestionAgent()
            docs_to_index = ingestion_agent.ingest(test_pdf_path)

            if docs_to_index:
                 retriever_agent = RetrieverAgent()
                 retriever_agent.index_knowledge(docs_to_index)

                 question = "Who is the author of the report?"
                 answer = retriever_agent.answer_question(question)
                 print("\n--- Final Answer ---")
                 print(answer)
                 print("--------------------")

                 question = "What is the VectorDB used for?"
                 answer = retriever_agent.answer_question(question)
                 print("\n--- Second Answer ---")
                 print(answer)
                 print("--------------------")

            if os.path.exists(test_pdf_path):
                 os.remove(test_pdf_path)

        except ImportError:
            print("\n❌ ReportLab not found. Skipping PDF test.")
            print("Install it using: pip install reportlab")
        except Exception as e:
            print(f"An error occurred during testing: {e}")
            # Clean up dummy file even if error occurs
            if 'test_pdf_path' in locals() and os.path.exists(test_pdf_path):
                 os.remove(test_pdf_path)

    except ImportError as e:
        print(f"Could not import IngestionAgent. Error: {e}")
        print("Make sure it's in app/agents/ingestion_agent.py")