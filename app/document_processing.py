"""Document loading and text chunking functionality."""

from pathlib import Path

import pypdf

from .config import config
from .models import DocumentChunk

logger = config.get_logger(__name__)


class DocumentLoader:
    """Handles loading of PDF and TXT documents."""

    @staticmethod
    def load_pdf(file_path: Path) -> str:
        """Load text content from a PDF file.

        Returns:
            The extracted text content from the PDF as a string.
        """
        try:
            with file_path.open("rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        except Exception:
            logger.exception("Error loading PDF %s", file_path)
            raise
        else:
            return text

    @staticmethod
    def load_txt(file_path: Path) -> str:
        """Load text content from a TXT file.

        Returns:
            The extracted text content from the TXT file as a string.
        """
        try:
            with file_path.open(encoding="utf-8") as file:
                text = file.read()
            logger.info("Successfully loaded TXT file")
        except Exception:
            logger.exception("Error loading TXT %s", file_path)
            raise
        else:
            return text

    @classmethod
    def load_document(cls, file_path: Path) -> str:
        """Load document based on file extension.

        Args:
            file_path: Path to the document file.

        Returns:
            The text content of the document as a string.

        Raises:
            ValueError: If the file type is not supported.
        """
        file_ext = file_path.suffix.lower()
        if file_ext == ".pdf":
            return cls.load_pdf(file_path)
        if file_ext == ".txt":
            return cls.load_txt(file_path)
        msg = f"Unsupported file type: {file_ext}"
        raise ValueError(msg)


class TextChunker:
    """Handles text chunking with fixed length and overlap strategy."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200) -> None:
        """Initialize the TextChunker with chunk size and overlap.

        Args:
            chunk_size: The size of each text chunk.
            overlap: The number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str = "document") -> list[DocumentChunk]:
        """Split text into overlapping chunks.

        Returns:
            A list of DocumentChunk objects representing the text chunks.
        """
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Ensure we don't break in the middle of a word (except for last chunk)
            if end < len(text) and not chunk_text.endswith(" "):
                last_space = chunk_text.rfind(" ")
                # At least half chunk size to prevent too small chunks after adjustment
                if last_space > start + self.chunk_size // 2:
                    end = start + last_space
                    chunk_text = text[start:end]

            if chunk_text.strip():  # Only add non-empty chunks
                chunk = DocumentChunk(
                    content=chunk_text.strip(),
                    metadata={
                        "source": source,
                        "chunk_id": chunk_id,
                        "start_char": start,
                        "end_char": end,
                        "length": len(chunk_text.strip()),
                    },
                )
                chunks.append(chunk)
                chunk_id += 1

            start = end - self.overlap

            if start >= end:
                break

        logger.info("Text split into %d chunks", len(chunks))
        return chunks
