# app/utils/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def chunk_text(text: str) -> list[str]:
    """
    Split text into smaller chunks using RecursiveCharacterTextSplitter
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# ----------------- Test -----------------
if __name__ == "__main__":
    sample_text = (
        "MemoryPalAI is an Agentic AI-powered second brain. "
        "It ingests notes, PDFs, audio, and bookmarks, organizes them, "
        "retrieves information, and builds learning roadmaps."
    )
    chunks = chunk_text(sample_text)
    print(f"Number of chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{c}")
