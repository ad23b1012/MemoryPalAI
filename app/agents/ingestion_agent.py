# app/agents/ingestion_agent.py
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from app.services.audio_service import transcribe_audio


class IngestionAgent:
    """
    Ingests documents from TXT, PDF, or audio files.
    Audio files are transcribed using Whisper.
    """

    def __init__(self):
        print("✅ IngestionAgent initialized.")

    def ingest(self, file_path: str):
        """
        Loads a document from the given file path.
        Handles text, PDF, and audio files (mp3, wav, m4a, mpeg).
        Returns a list of LangChain Document objects or None on failure.
        """
        # ---------- Input Validation ----------
        if not file_path or not os.path.exists(file_path):
            print(f"❌ Invalid or missing file path: {file_path}")
            return None

        _, ext = os.path.splitext(file_path.lower())
        print(f"🚀 IngestionAgent: processing file '{file_path}'")

        audio_exts = ['.mp3', '.m4a', '.wav', '.mpeg']

        try:
            # ---------- TXT Files ----------
            if ext == ".txt":
                loader = TextLoader(file_path)
                docs = loader.load()

            # ---------- PDF Files ----------
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()

            # ---------- Audio Files ----------
            elif ext in audio_exts:
                print("🎧 Audio file detected — transcribing via Whisper...")
                transcribed_text = transcribe_audio(file_path)

                if not transcribed_text or "Error" in transcribed_text:
                    print(f"❌ Transcription failed for: {file_path}")
                    return None

                docs = [Document(page_content=transcribed_text, metadata={"source": file_path})]

            # ---------- Unsupported ----------
            else:
                print(f"⚠️ Unsupported file type: {ext}")
                return None

            print(f"✅ Ingestion complete. Processed {len(docs)} document(s).")
            return docs

        except Exception as e:
            print(f"❌ Error during ingestion: {e}")
            return None


# ---------------- Test Block ----------------
if __name__ == "__main__":
    # Create a small test file to verify ingestion logic
    test_path = "test_note.txt"
    with open(test_path, "w") as f:
        f.write("Artificial Intelligence enables machines to think and learn like humans.")

    agent = IngestionAgent()
    result = agent.ingest(test_path)

    if result:
        print("\n--- Ingested Document Content ---")
        print(result[0].page_content)
        print("-------------------------------")
    else:
        print("❌ Ingestion failed.")

    # Clean up
    os.remove(test_path)
