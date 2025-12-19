"""FAISS-backed vector storage with SQLite metadata."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from app.config import config
from app.vector_store.base import DEFAULT_DOC_TYPE, BaseSQLiteStore

if TYPE_CHECKING:
    from app.models import DocumentChunk

logger = config.get_logger(__name__)


class FaissVectorStore(BaseSQLiteStore):
    """Vector storage using FAISS for embeddings and SQLite for metadata."""

    backend = "faiss"

    def __init__(
        self,
        db_path: Path = Path("data/vector_store.db"),
        index_path: Path = Path("data/faiss/index.faiss"),
        raw_top_k_multiplier: int = 2,
    ) -> None:
        """Configure FAISS-backed vector store."""
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(exist_ok=True, parents=True)

        self.index: faiss.IndexIDMap | None = None
        self.embeddings: None = None  # Compatibility placeholder
        self.raw_top_k_multiplier = max(1, raw_top_k_multiplier)
        self.chunks: list[DocumentChunk] = []

        super().__init__(db_path)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity using inner product search.

        Returns:
            Normalized embedding vector.
        """
        vector = np.asarray(embedding, dtype="float32")
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        faiss.normalize_L2(vector.reshape(1, -1))
        return vector

    def _init_index(self, dimension: int) -> None:
        """Initialize FAISS index if missing."""
        base_index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap(base_index)
        logger.info("Initialized FAISS IndexIDMap with dimension %d", dimension)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks and embeddings to FAISS index and metadata store.

        Raises:
            ValueError: If embedding dimension mismatches the index.
            RuntimeError: If the FAISS index cannot store provided ids.
        """
        if not chunks:
            return

        embeddings_batch: list[np.ndarray] = []
        vector_ids: list[int] = []

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(
                        "Skipping chunk %s without embedding",
                        chunk.metadata.get("chunk_id"),
                    )
                    continue

                embedding = self._normalize_embedding(np.asarray(chunk.embedding))
                if self.index is None:
                    self._init_index(embedding.shape[0])
                elif embedding.shape[0] != self.index.d:
                    msg = (
                        f"Embedding dimension {embedding.shape[0]} does not match "
                        f"FAISS index dimension {self.index.d}"
                    )
                    raise ValueError(msg)

                source = chunk.metadata.get("source", "unknown")
                document_id = self._upsert_document(cursor, source)

                _chunk_db_id, vector_id, labels = self._insert_chunk_row(
                    cursor,
                    document_id,
                    chunk,
                    vector_file=None,
                )

                chunk.metadata.setdefault("vector_id", vector_id)
                chunk.metadata.setdefault("doc_type", DEFAULT_DOC_TYPE)
                if labels:
                    chunk.metadata["labels"] = labels

                embeddings_batch.append(embedding)
                vector_ids.append(vector_id)
                self.chunks.append(chunk)

            conn.commit()

        if embeddings_batch and self.index is not None:
            vectors = np.vstack(embeddings_batch).astype("float32")
            ids_array = np.asarray(vector_ids, dtype="int64")
            try:
                self.index.add_with_ids(vectors, ids_array)  # pyright: ignore[reportCallIssue]  # FAISS stubs may not reflect add_with_ids signature
            except RuntimeError:
                logger.exception(
                    "FAISS index does not support add_with_ids; "
                    "ensure IndexIDMap is used."
                )
                raise
            logger.info("Added %d vectors to FAISS index", len(vector_ids))
        else:
            logger.warning("No embeddings added to FAISS index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search similar chunks using FAISS index.

        Returns:
            Ranked list of (DocumentChunk, score) tuples.
        """
        index = self.index
        if index is None:
            if self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                self.index = index
                logger.info(
                    "Loaded FAISS index from disk with %d vectors", index.ntotal
                )
            else:
                logger.warning("FAISS index not initialized; returning no results")
                return []

        if index.ntotal == 0:
            return []

        normalized_query = self._normalize_embedding(np.asarray(query_embedding))
        raw_top_k = max(top_k, self.raw_top_k_multiplier * top_k)
        raw_top_k = min(raw_top_k, index.ntotal)

        scores, vector_ids = index.search(
            normalized_query.reshape(1, -1),
            raw_top_k,
        )  # pyright: ignore[reportCallIssue]

        results: list[tuple[DocumentChunk, float]] = []
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for score, vector_id in zip(scores[0], vector_ids[0], strict=True):
                if int(vector_id) == -1:  # faiss returns -1 for empty results
                    continue
                chunk = self._fetch_chunk_by_vector_id(cursor, int(vector_id))
                if chunk:
                    results.append((chunk, float(score)))

        return results[:top_k]

    def save(self) -> None:
        """Persist FAISS index to disk."""
        index = self.index
        if index is None:
            logger.warning("No FAISS index to save")
            return

        self.index_path.parent.mkdir(exist_ok=True, parents=True)
        faiss.write_index(index, str(self.index_path))
        logger.info("Saved FAISS index to %s", self.index_path)

    def load(self) -> None:
        """Load metadata and FAISS index from disk.

        Raises:
            sqlite3.Error: If metadata read fails.
        """
        if self.index_path.exists():
            loaded_index = faiss.read_index(str(self.index_path))
            self.index = loaded_index
            logger.info(
                "Loaded FAISS index from %s with %d vectors",
                self.index_path,
                loaded_index.ntotal,
            )
        else:
            logger.warning(
                "FAISS index not found at %s. Start with an empty index.",
                self.index_path,
            )
            self.index = None

        index = self.index
        if index is not None and not isinstance(
            index, (faiss.IndexIDMap, faiss.IndexIDMap2)
        ):
            logger.warning(
                "Loaded FAISS index is %s; wrapping with IndexIDMap to enable IDs",
                type(index).__name__,
            )
            self.index = faiss.IndexIDMap(index)

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        c.id,
                        c.content,
                        c.start_char,
                        c.end_char,
                        c.length,
                        c.chunk_id,
                        c.vector_file,
                        c.vector_id,
                        c.doc_type,
                        c.topic,
                        c.created_at,
                        c.user,
                        d.source
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    ORDER BY c.id
                """)

                self.chunks = []
                for row in cursor.fetchall():
                    chunk_db_id = int(row[0])
                    labels = self._load_labels(cursor, chunk_db_id)
                    chunk = self._build_chunk_from_row(row)
                    if labels:
                        chunk.metadata["labels"] = labels
                    self.chunks.append(chunk)

            logger.info("Loaded %d chunks from metadata store", len(self.chunks))

        except sqlite3.Error:
            logger.exception("Error loading metadata for FAISS vector store")
            raise
