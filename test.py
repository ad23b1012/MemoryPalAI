import asyncio
from app.services.parser import document_parser
from app.utils.chunker import chunk_from_text

async def main():
    url = "https://arxiv.org/pdf/2303.00001.pdf"  # Example PDF
    text = await document_parser.parse_document(url)
    if not text:
        print("Failed to parse document.")
        return

    print(f"Document length: {len(text)} characters")
    chunks = await chunk_from_text(text)
    print(f"Number of chunks: {len(chunks)}")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{c[:200]}...")  # print first 200 chars

asyncio.run(main())
