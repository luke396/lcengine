"""SQLite-based vector storage with numpy file backend."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from app.config import config
from app.models import DocumentChunk  # noqa: TC001
from app.vector_store.base import DEFAULT_DOC_TYPE, BaseSQLiteStore

logger = config.get_logger(__name__)


class SQLiteVectorStore(BaseSQLiteStore):
    """Vector storage using SQLite for metadata and numpy files for embeddings."""

    backend = "sqlite"

    def __init__(
        self,
        db_path: Path = Path("data/vector_store.db"),
        vectors_dir: Path = Path("data/vectors"),
    ) -> None:
        """Initialize the SQLiteVectorStore with database and vector directory paths.

        Args:
            db_path: Path to the SQLite database file.
            vectors_dir: Directory to store numpy vector files.
        """
        self.vectors_dir = Path(vectors_dir)
        self.vectors_dir.mkdir(exist_ok=True, parents=True)

        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

        super().__init__(db_path)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks with embeddings to the store."""
        if not chunks:
            return

        inserted = 0

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(
                        "Skipping chunk %s without embedding",
                        chunk.metadata.get("chunk_id"),
                    )
                    continue

                source = chunk.metadata.get("source", "unknown")
                document_id = self._upsert_document(cursor, source)

                chunk_id = chunk.metadata.get("chunk_id", 0)
                vector_filename = f"doc{document_id:06d}_chunk{chunk_id:06d}.npy"
                vector_path = self.vectors_dir / vector_filename
                np.save(vector_path, chunk.embedding)

                _chunk_db_id, vector_id, labels = self._insert_chunk_row(
                    cursor,
                    document_id,
                    chunk,
                    vector_file=str(vector_filename),
                )

                chunk.metadata.setdefault("vector_id", vector_id)
                chunk.metadata.setdefault("doc_type", DEFAULT_DOC_TYPE)
                if labels:
                    chunk.metadata["labels"] = labels

                self.chunks.append(chunk)
                inserted += 1

            conn.commit()

        self._rebuild_embeddings_matrix()

        logger.info("Added %d chunks to SQLite vector store", inserted)

    def _rebuild_embeddings_matrix(self) -> None:
        """Rebuild the embeddings matrix from individual vector files."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT vector_file FROM chunks
                WHERE vector_file IS NOT NULL
                ORDER BY id
                """
            )
            vector_files = [row[0] for row in cursor.fetchall()]

        if not vector_files:
            self.embeddings = None
            return

        embeddings_list = []
        for vector_file in vector_files:
            vector_path = self.vectors_dir / vector_file
            if vector_path.exists():
                embedding = np.load(vector_path)
                embeddings_list.append(embedding)
            else:
                logger.warning("Vector file not found: %s", vector_path)

        if embeddings_list:
            self.embeddings = np.vstack(embeddings_list)
        else:
            self.embeddings = None

        logger.info("Rebuilt embeddings matrix with %d vectors", len(embeddings_list))

    @staticmethod
    def cosine_similarity(
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Calculate cosine similarity between query and document embeddings.

        Returns:
            np.ndarray: Array of cosine similarity scores
                    between the query and each document embedding.
        """
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return np.dot(doc_norms, query_norm)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for similar chunks based on query embedding.

        Returns:
            A list of tuples, each containing a DocumentChunk and its similarity score.
        """
        if self.embeddings is None:
            self._rebuild_embeddings_matrix()

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        similarities = self.cosine_similarity(query_embedding, self.embeddings)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for idx in top_indices:
                score = similarities[idx]
                chunk = self._fetch_chunk_by_offset(cursor, idx)
                if chunk:
                    logger.info(
                        "Retrieved chunk %s with similarity %.4f",
                        chunk.metadata.get("chunk_id"),
                        score,
                    )
                    results.append((chunk, float(score)))

        return results

    def save(self) -> None:  # noqa: PLR6301
        """
        Save operation - data is already persisted in SQLite and files.

        Note:
            This method is kept as an instance method for interface consistency
            with other vector store implementations, even though it does not use `self`.
        """
        logger.info("Data already persisted in SQLite database and vector files")

    def load(self) -> None:
        """Load chunks from SQLite database.

        Raises:
            sqlite3.Error: If an error occurs while loading
                    from the SQLite vector store.
        """
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

                    vector_file = chunk.metadata.get("vector_file")
                    if vector_file:
                        vector_path = self.vectors_dir / str(vector_file)
                        if vector_path.exists():
                            chunk.embedding = np.load(vector_path)
                        else:
                            logger.warning("Vector file missing: %s", vector_path)

                    self.chunks.append(chunk)

            self._rebuild_embeddings_matrix()

            logger.info("Loaded %d chunks from SQLite vector store", len(self.chunks))

        except sqlite3.Error:
            logger.exception("Error loading from SQLite vector store")
            raise

    def _load_embedding(self, vector_path: Path) -> np.ndarray | None:
        """Load a numpy embedding from disk.

        Returns:
            Numpy array if present on disk; otherwise None.
        """
        resolved_path = (
            vector_path if vector_path.is_absolute() else self.vectors_dir / vector_path
        )
        if not resolved_path.exists():
            logger.warning("Vector file not found: %s", resolved_path)
            return None
        return np.load(resolved_path)
