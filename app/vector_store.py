"""SQLite-based vector storage with numpy file backend."""

import sqlite3
from pathlib import Path

import numpy as np

from .config import config
from .models import DocumentChunk

logger = config.get_logger(__name__)


class SQLiteVectorStore:
    """Vector storage using SQLite for metadata and numpy files for embeddings."""

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
        self.db_path = db_path
        self.vectors_dir = vectors_dir
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self.vectors_dir.mkdir(exist_ok=True, parents=True)

        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL UNIQUE,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    start_char INTEGER,
                    end_char INTEGER,
                    length INTEGER,
                    vector_file TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
            )
            cursor.execute(
                (
                    "CREATE INDEX IF NOT EXISTS idx_chunks_source "
                    "ON chunks(document_id, chunk_id)"
                ),
            )

            conn.commit()
            logger.info("Database tables created/verified")

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks with embeddings to the store."""
        if not chunks:
            return

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

                cursor.execute(
                    "INSERT OR IGNORE INTO documents (source) VALUES (?)",
                    (source,),
                )
                cursor.execute("SELECT id FROM documents WHERE source = ?", (source,))
                doc_id = cursor.fetchone()[0]

                chunk_id = chunk.metadata.get("chunk_id", 0)
                vector_filename = f"doc{doc_id:06d}_chunk{chunk_id:06d}.npy"
                vector_path = self.vectors_dir / vector_filename
                np.save(vector_path, chunk.embedding)

                cursor.execute(
                    """
                    INSERT INTO chunks
                    (
                        document_id,
                        chunk_id,
                        content,
                        start_char,
                        end_char,
                        length,
                        vector_file
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        doc_id,
                        chunk.metadata.get("chunk_id", 0),
                        chunk.content,
                        chunk.metadata.get("start_char", 0),
                        chunk.metadata.get("end_char", 0),
                        chunk.metadata.get("length", len(chunk.content)),
                        vector_filename,
                    ),
                )

                self.chunks.append(chunk)

            conn.commit()

        self._rebuild_embeddings_matrix()

        logger.info("Added %d chunks to SQLite vector store", len(chunks))

    def _rebuild_embeddings_matrix(self) -> None:
        """Rebuild the embeddings matrix from individual vector files."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vector_file FROM chunks ORDER BY id")
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

    def _fetch_chunk_by_index(
        self, idx: int, cursor: sqlite3.Cursor
    ) -> DocumentChunk | None:
        cursor.execute(
            """
            SELECT
                c.content,
                c.start_char,
                c.end_char,
                c.length,
                c.chunk_id,
                c.vector_file,
                d.source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY c.id
            LIMIT 1 OFFSET ?
            """,
            (int(idx),),
        )
        row = cursor.fetchone()
        if row:
            (
                content,
                start_char,
                end_char,
                length,
                chunk_id,
                vector_file,
                source,
            ) = row

            vector_path = self.vectors_dir / vector_file
            embedding = np.load(vector_path) if vector_path.exists() else None

            return DocumentChunk(
                content=content,
                metadata={
                    "source": source,
                    "chunk_id": chunk_id,
                    "start_char": start_char,
                    "end_char": end_char,
                    "length": length,
                },
                embedding=embedding,
            )

        return None

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Search for similar chunks based on query embedding.

        Returns:
            A list of tuples, each containing a DocumentChunk and its similarity score.
        """
        # TODO(luke): Limit the similarity score # noqa: FIX002
        # to filter out irrelevant results.
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
                chunk = self._fetch_chunk_by_index(idx, cursor)
                if chunk:
                    logger.info(
                        "Retrieved chunk %s with similarity %.4f",
                        chunk.metadata.get("chunk_id"),
                        score,
                    )
                    results.append((chunk, float(score)))

        return results

    @staticmethod
    def save() -> None:
        """Save operation - data is already persisted in SQLite and files."""
        # not actually save here, we save on add_chunks
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
                    SELECT c.content, c.start_char, c.end_char, c.length, c.chunk_id,
                           c.vector_file, d.source
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    ORDER BY c.id
                """)

                self.chunks = []
                for row in cursor.fetchall():
                    (
                        content,
                        start_char,
                        end_char,
                        length,
                        chunk_id,
                        vector_file,
                        source,
                    ) = row

                    vector_path = self.vectors_dir / vector_file
                    embedding = np.load(vector_path) if vector_path.exists() else None

                    chunk = DocumentChunk(
                        content=content,
                        metadata={
                            "source": source,
                            "chunk_id": chunk_id,
                            "start_char": start_char,
                            "end_char": end_char,
                            "length": length,
                        },
                        embedding=embedding,
                    )
                    self.chunks.append(chunk)

            self._rebuild_embeddings_matrix()

            logger.info("Loaded %d chunks from SQLite vector store", len(self.chunks))

        except sqlite3.Error:
            logger.exception("Error loading from SQLite vector store")
            raise
