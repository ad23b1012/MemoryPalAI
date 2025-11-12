# app/agents/ingestion_agent.py
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from app.services.audio_service import transcribe_audio  # keep your existing audio_service

class IngestionAgent:
    """
    Ingests TXT, PDF, or audio files and returns a list of Document objects with metadata.
    """

    def __init__(self):
        print("✅ IngestionAgent initialized.")

    def ingest(self, file_path: str):
        if not file_path or not os.path.exists(file_path):
            print(f"❌ Invalid or missing file path: {file_path}")
            return None

        _, ext = os.path.splitext(file_path.lower())
        print(f"🚀 IngestionAgent: processing file '{file_path}'")

        try:
            if ext == ".txt":
                loader = TextLoader(file_path)
                docs = loader.load()

            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()

            elif ext in [".mp3", ".m4a", ".wav", ".mpeg"]:
                print("🎧 Audio detected — transcribing...")
                transcribed = transcribe_audio(file_path)
                if not transcribed:
                    print("❌ Transcription returned nothing.")
                    return None
                docs = [Document(page_content=transcribed, metadata={"source": file_path})]

            else:
                print(f"⚠️ Unsupported file type: {ext}")
                return None

            # Ensure metadata present for each doc
            for i, d in enumerate(docs):
                if not getattr(d, "metadata", None):
                    d.metadata = {"source": file_path, "page": i}
                else:
                    d.metadata.setdefault("source", file_path)
                    d.metadata.setdefault("page", i)

            print(f"✅ Ingestion complete. Processed {len(docs)} document(s).")
            return docs

        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            return None
