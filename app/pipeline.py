"""Main RAG pipeline orchestrating document processing and querying."""

from pathlib import Path

from .config import config
from .document_processing import DocumentLoader, TextChunker
from .embeddings import EmbeddingService
from .models import DocumentChunk
from .vector_store import SQLiteVectorStore

logger = config.get_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrating Load -> Split -> Embed -> Store."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        sqlite_db_path: Path | None = None,
        vectors_dir: Path | None = None,
    ) -> None:
        """Initialize RAG pipeline with SQLite + numpy storage.

        Args:
            openai_api_key: OpenAI API key.
            chunk_size: Size of text chunks. If None, uses config.CHUNK_SIZE.
            overlap: Overlap between chunks. If None, uses config.CHUNK_OVERLAP.
            sqlite_db_path: Path for SQLite database.
                    If None, uses config.VECTOR_STORE_DB_PATH.
            vectors_dir: Directory for numpy vector files.
                    If None, uses config.VECTOR_STORE_DIR.
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP
        sqlite_db_path = sqlite_db_path or config.VECTOR_STORE_DB_PATH
        vectors_dir = vectors_dir or config.VECTOR_STORE_DIR

        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_service = EmbeddingService(api_key=openai_api_key)

        # Use SQLite + numpy storage
        self.vector_store = SQLiteVectorStore(
            db_path=sqlite_db_path,
            vectors_dir=vectors_dir,
        )
        logger.info("Using SQLite + numpy vector storage")

        self.vector_store.load()

    def process_document(self, file_path: Path) -> None:
        """Process a document through the complete RAG pipeline."""
        logger.info("Starting RAG pipeline for document: %s", file_path)

        text = DocumentLoader.load_document(file_path)
        chunks = self.chunker.chunk_text(text, source=file_path.name)

        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.get_embeddings_batch(chunk_texts)

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding

        self.vector_store.add_chunks(chunks)
        self.vector_store.save()

        logger.info("Document processing completed successfully")

    def query(self, question: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Query the RAG system.

        Args:
            question: The input question to query.
            top_k: Number of top results to return.

        Returns:
            A list of tuples, each containing a DocumentChunk and its similarity score.
        """
        logger.info("Processing query: %s", question)

        query_embedding = self.embedding_service.get_embedding(question)

        return self.vector_store.search(query_embedding, top_k=top_k)
