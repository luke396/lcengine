"""Main RAG pipeline orchestrating document processing and querying."""

from pathlib import Path
from typing import cast

from .config import config
from .document_processing import DocumentLoader, TextChunker
from .embeddings import EmbeddingService
from .models import DocumentChunk
from .vector_store import VectorBackend, get_vector_store

logger = config.get_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrating Load -> Split -> Embed -> Store."""

    def __init__(  # noqa: PLR0913,PLR0917
        self,
        openai_api_key: str | None = None,
        chunk_size: int | None = None,
        overlap: int | None = None,
        sqlite_db_path: Path | None = None,
        vectors_dir: Path | None = None,
        vector_backend: str | None = None,
        faiss_index_path: Path | None = None,
    ) -> None:
        """Initialize RAG pipeline with configurable vector storage.

        Args:
            openai_api_key: OpenAI API key.
            chunk_size: Size of text chunks. If None, uses config.CHUNK_SIZE.
            overlap: Overlap between chunks. If None, uses config.CHUNK_OVERLAP.
            sqlite_db_path: Path for SQLite database/metadata. If None, uses
                config.VECTOR_STORE_DB_PATH.
            vectors_dir: Directory for numpy vector files (SQLite backend).
                If None, uses config.VECTOR_STORE_DIR.
            vector_backend: Which vector store backend to use ("faiss" | "sqlite").
                Defaults to config.VECTOR_BACKEND.
            faiss_index_path: Path to FAISS index file. If None, uses
                config.FAISS_INDEX_PATH.
        """
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if overlap is None:
            overlap = config.CHUNK_OVERLAP
        if sqlite_db_path is None:
            sqlite_db_path = config.VECTOR_STORE_DB_PATH
        if vectors_dir is None:
            vectors_dir = config.VECTOR_STORE_DIR
        if faiss_index_path is None:
            faiss_index_path = config.FAISS_INDEX_PATH
        backend_value = (
            vector_backend if vector_backend is not None else config.VECTOR_BACKEND
        )
        backend = cast("VectorBackend", backend_value.lower())

        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedding_service = EmbeddingService(api_key=openai_api_key)

        self.vector_store = get_vector_store(
            backend,
            db_path=sqlite_db_path,
            vectors_dir=vectors_dir,
            index_path=faiss_index_path,
        )
        self.vector_backend = getattr(self.vector_store, "backend", backend)
        logger.info("Using %s vector storage", self.vector_backend)

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
