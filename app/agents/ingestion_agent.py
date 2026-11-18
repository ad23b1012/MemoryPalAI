# app/agents/ingestion_agent.py
import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from app.services.audio_service import transcribe_audio

logger = logging.getLogger(__name__)


class IngestionAgent:
    """
    Ingests TXT, MD, PDF, DOCX, audio files (MP3, WAV, M4A) or URLs,
    returning structured LangChain Document objects with rich metadata.
    """

    def __init__(self):
        logger.info("IngestionAgent initialized.")

    def ingest(self, file_path: str) -> Optional[List[Document]]:
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Invalid or missing file path: {file_path}")
            return None

        _, ext = os.path.splitext(file_path.lower())
        logger.info(f"IngestionAgent processing: {file_path}")

        try:
            docs: List[Document] = []

            if ext in [".txt", ".md", ".csv", ".json"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                docs = [Document(page_content=content, metadata={"source": os.path.basename(file_path)})]

            elif ext == ".pdf":
                # Try PyMuPDF (fitz) first for fastest and best extraction, fallback to pypdf
                try:
                    import fitz
                    pdf_doc = fitz.open(file_path)
                    pages_text = []
                    for idx, page in enumerate(pdf_doc):
                        t = page.get_text()
                        if t.strip():
                            pages_text.append(t)
                    combined_text = "\n\n".join(pages_text)
                    docs = [Document(page_content=combined_text, metadata={"source": os.path.basename(file_path), "total_pages": len(pdf_doc)})]
                except Exception as e:
                    logger.warning(f"PyMuPDF failed ({e}), trying PyPDFLoader...")
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

            elif ext in [".docx", ".doc"]:
                try:
                    import docx
                    doc_obj = docx.Document(file_path)
                    full_text = "\n".join([p.text for p in doc_obj.paragraphs if p.text.strip()])
                    docs = [Document(page_content=full_text, metadata={"source": os.path.basename(file_path)})]
                except Exception as e:
                    logger.error(f"Failed to read DOCX: {e}")
                    return None

            elif ext in [".mp3", ".m4a", ".wav", ".mpeg", ".aac", ".ogg"]:
                logger.info("Audio format detected. Transcribing with Whisper...")
                transcribed = transcribe_audio(file_path)
                if not transcribed or transcribed.startswith("❌"):
                    logger.error("Transcription returned no usable content.")
                    return None
                docs = [Document(page_content=transcribed, metadata={"source": os.path.basename(file_path), "media": "audio"})]

            else:
                logger.warning(f"Unsupported file format: {ext}")
                return None

            # Standardize metadata
            for i, d in enumerate(docs):
                d.metadata = d.metadata or {}
                d.metadata.setdefault("source", os.path.basename(file_path))
                d.metadata.setdefault("chunk_index", i)

            logger.info(f"Ingested {len(docs)} document chunk(s) from {file_path}")
            return docs

        except Exception as e:
            logger.error(f"Error during file ingestion ({file_path}): {e}")
            return None
