from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents: list[Document], chunk_size=1000, chunk_overlap=100) -> list[Document]:
    """
    Splits a list of LangChain documents into smaller chunks.
    """
    print(f"Chunking {len(documents)} document(s)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use standard character length
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✅ Documents split into {len(split_docs)} chunks.")
    return split_docs

if __name__ == "__main__":
    # Test block for the chunker
    print("--- Testing Chunker Utility ---")
    # Create a dummy long document
    long_text = "This is sentence one. " * 300 # Approx 1500 chars
    test_doc = Document(page_content=long_text, metadata={"source": "test"})
    chunks = chunk_documents([test_doc])
    print(f"First chunk content preview: '{chunks[0].page_content[:100]}...'")
    print(f"Number of chunks created: {len(chunks)}")