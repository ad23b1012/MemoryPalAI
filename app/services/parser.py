# app/services/parser.py
import io
import httpx
import logging
from typing import Optional, Tuple
from urllib.parse import urlparse
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from email import message_from_bytes
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DocumentParser:
    async def _download_document(self, url: str) -> Optional[Tuple[bytes, str]]:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                return response.content, content_type
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None

    def _parse_pdf(self, content: bytes) -> Optional[str]:
        try:
            reader = PdfReader(io.BytesIO(content))
            return "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.error(f"PDF parse error: {e}")
            return None

    def _parse_docx(self, content: bytes) -> Optional[str]:
        try:
            doc = DocxDocument(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            logger.error(f"DOCX parse error: {e}")
            return None

    def _parse_eml(self, content: bytes) -> Optional[str]:
        try:
            msg = message_from_bytes(content)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype in ["text/plain", "text/html"]:
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        if ctype == "text/html":
                            body = BeautifulSoup(body, "html.parser").get_text()
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")
            return body
        except Exception as e:
            logger.error(f"EML parse error: {e}")
            return None

    def _infer_extension(self, url: str, content_type: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.lower()
        if ".pdf" in path or "application/pdf" in content_type:
            return "pdf"
        if ".docx" in path or "wordprocessingml" in content_type:
            return "docx"
        if ".eml" in path or "message/rfc822" in content_type:
            return "eml"
        return "unknown"

    async def parse_document(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        result = await self._download_document(url)
        if not result:
            return None, None
        content, content_type = result
        file_type = self._infer_extension(url, content_type)
        if file_type == "pdf":
            return self._parse_pdf(content), "application/pdf"
        elif file_type == "docx":
            return self._parse_docx(content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif file_type == "eml":
            return self._parse_eml(content), "message/rfc822"
        return None, None


document_parser = DocumentParser()


# ----------------- Test -----------------
if __name__ == "__main__":
    import asyncio
    async def test():
        # Replace with any online PDF or DOCX for testing
        url = "https://arxiv.org/pdf/2303.00001.pdf"
        text, mime = await document_parser.parse_document(url)
        if text:
            print(f"Document type: {mime}")
            print(f"Document length: {len(text)} chars")
        else:
            print("Failed to parse document")

    asyncio.run(test())
