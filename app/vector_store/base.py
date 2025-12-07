"""Shared helpers for SQLite-backed vector stores."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from app.config import config
from app.models import DocumentChunk

DEFAULT_DOC_TYPE = "document"
VALID_DOC_TYPES = {"document", "note", "mistake"}

logger = config.get_logger(__name__)


class BaseSQLiteStore:
    """Common schema management and helpers for vector stores using SQLite metadata."""

    def __init__(self, db_path: Path) -> None:
        """Initialize metadata store and ensure schema exists."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self._schema_upgraded = False
        self._create_tables()

    def _create_tables(self) -> None:
        """Create metadata and labels tables if they don't exist."""
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
                    vector_file TEXT,
                    vector_id INTEGER UNIQUE,
                    doc_type TEXT DEFAULT 'document' CHECK(
                        doc_type IN ('document','note','mistake')
                    ),
                    topic TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)

            added_columns = self._ensure_chunk_columns(cursor)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS labels (
                    chunk_id INTEGER,
                    label TEXT
                )
            """)

            self._create_indexes(cursor)
            conn.commit()

            if added_columns:
                logger.warning(
                    "Updated vector store schema with columns: %s. "
                    "If you previously ingested data, consider rebuilding indexes.",
                    ", ".join(sorted(added_columns)),
                )
            self._schema_upgraded = bool(added_columns)

    @staticmethod
    def _ensure_chunk_columns(cursor: sqlite3.Cursor) -> set[str]:
        """Add new metadata columns if missing for backward compatibility.

        Returns:
            Set of column names that were added during migration.
        """
        cursor.execute("PRAGMA table_info(chunks)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        added_columns: set[str] = set()

        required_columns: dict[str, str] = {
            "vector_file": "TEXT",
            "vector_id": "INTEGER UNIQUE",
            "doc_type": (
                "TEXT DEFAULT 'document' "
                "CHECK(doc_type IN ('document','note','mistake'))"
            ),
            "topic": "TEXT",
            "created_at": "DATETIME DEFAULT CURRENT_TIMESTAMP",
            "user": "TEXT",
        }

        for column_name, ddl in required_columns.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {ddl}")
                added_columns.add(column_name)

        cursor.execute("UPDATE chunks SET vector_id = id WHERE vector_id IS NULL")
        cursor.execute(
            "UPDATE chunks SET doc_type = ? WHERE doc_type IS NULL",
            (DEFAULT_DOC_TYPE,),
        )
        cursor.execute(
            "UPDATE chunks SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
        )

        return added_columns

    @staticmethod
    def _create_indexes(cursor: sqlite3.Cursor) -> None:
        """Ensure metadata indexes exist for common filters."""
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
        )
        cursor.execute(
            (
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_vector_id "
                "ON chunks(vector_id)"
            ),
        )
        cursor.execute(
            (
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type_topic "
                "ON chunks(doc_type, topic)"
            ),
        )
        cursor.execute(
            (
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type_created_at "
                "ON chunks(doc_type, created_at DESC)"
            ),
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_label ON labels(label)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_labels_chunk ON labels(chunk_id)"
        )

    @staticmethod
    def _normalize_doc_type(doc_type: str | None) -> str:
        """Sanitize doc_type to a supported value.

        Returns:
            A supported doc_type string.
        """
        if not doc_type or doc_type not in VALID_DOC_TYPES:
            return DEFAULT_DOC_TYPE
        return doc_type

    @staticmethod
    def _normalize_labels(labels: object) -> list[str]:
        """Convert labels to a normalized list of strings.

        Returns:
            Deduplicated list of string labels.
        """
        if labels is None:
            return []
        if isinstance(labels, str):
            return [labels]
        if isinstance(labels, Iterable):
            normalized = [str(label) for label in labels if str(label).strip()]
            return list(dict.fromkeys(normalized))
        return []

    def _normalize_metadata_fields(
        self,
        metadata: dict[str, Any] | None,
    ) -> tuple[str, str | None, str | None, list[str], str | None, int | None]:
        """Extract normalized metadata values with defaults.

        Returns:
            Tuple of (doc_type, topic, user, labels, created_at, vector_id).
        """
        metadata = metadata or {}
        doc_type = self._normalize_doc_type(
            str(metadata.get("doc_type", DEFAULT_DOC_TYPE))
        )
        topic = metadata.get("topic")
        user = metadata.get("user")
        labels = self._normalize_labels(metadata.get("labels"))
        created_at = metadata.get("created_at")
        vector_id_raw = metadata.get("vector_id")
        vector_id = int(vector_id_raw) if vector_id_raw is not None else None

        return doc_type, topic, user, labels, created_at, vector_id

    @staticmethod
    def _upsert_document(cursor: sqlite3.Cursor, source: str) -> int:
        """Insert document metadata if missing and return its id.

        Raises:
            RuntimeError: If the document id cannot be retrieved.

        Returns:
            Document id from the metadata store.
        """
        cursor.execute("INSERT OR IGNORE INTO documents (source) VALUES (?)", (source,))
        cursor.execute("SELECT id FROM documents WHERE source = ?", (source,))
        row = cursor.fetchone()
        if row is None:
            msg = f"Failed to upsert document for source '{source}'"
            raise RuntimeError(msg)
        return int(row[0])

    @staticmethod
    def _replace_labels(
        cursor: sqlite3.Cursor,
        chunk_db_id: int,
        labels: list[str],
    ) -> None:
        """Replace labels for a chunk."""
        cursor.execute("DELETE FROM labels WHERE chunk_id = ?", (chunk_db_id,))
        for label in labels:
            cursor.execute(
                "INSERT INTO labels (chunk_id, label) VALUES (?, ?)",
                (chunk_db_id, label),
            )

    @staticmethod
    def _load_labels(cursor: sqlite3.Cursor, chunk_db_id: int) -> list[str]:
        """Fetch labels for a chunk.

        Returns:
            Sorted list of labels tied to the chunk.
        """
        cursor.execute(
            "SELECT label FROM labels WHERE chunk_id = ? ORDER BY label",
            (chunk_db_id,),
        )
        return [row[0] for row in cursor.fetchall()]

    def _insert_chunk_row(
        self,
        cursor: sqlite3.Cursor,
        document_id: int,
        chunk: DocumentChunk,
        *,
        vector_file: str | None,
    ) -> tuple[int, int, list[str]]:
        """Persist a chunk row and return (chunk_db_id, vector_id, labels).

        Raises:
            RuntimeError: If the chunk row cannot be inserted.

        Returns:
            Tuple of row id, vector id, and normalized labels list.
        """
        doc_type, topic, user, labels, created_at, vector_id = (
            self._normalize_metadata_fields(chunk.metadata)
        )

        cursor.execute(
            """
            INSERT INTO chunks (
                document_id,
                chunk_id,
                content,
                start_char,
                end_char,
                length,
                vector_file,
                vector_id,
                doc_type,
                topic,
                user,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            """,
            (
                document_id,
                chunk.metadata.get("chunk_id", 0),
                chunk.content,
                chunk.metadata.get("start_char", 0),
                chunk.metadata.get("end_char", 0),
                chunk.metadata.get("length", len(chunk.content)),
                vector_file,
                vector_id,
                doc_type,
                topic,
                user,
                created_at,
            ),
        )

        chunk_row_id = cursor.lastrowid
        if chunk_row_id is None:
            msg = "Failed to insert chunk row"
            raise RuntimeError(msg)
        chunk_db_id = int(chunk_row_id)
        assigned_vector_id = int(vector_id) if vector_id is not None else chunk_db_id
        cursor.execute(
            "UPDATE chunks SET vector_id = ? WHERE id = ?",
            (assigned_vector_id, chunk_db_id),
        )
        self._replace_labels(cursor, chunk_db_id, labels)
        return chunk_db_id, assigned_vector_id, labels

    def _build_chunk_from_row(
        self,
        row: tuple,
        *,
        embedding: np.ndarray | None = None,
    ) -> DocumentChunk:
        """Create a DocumentChunk from a metadata row.

        Returns:
            DocumentChunk hydrated with metadata and optional embedding.
        """
        (
            chunk_db_id,
            content,
            start_char,
            end_char,
            length,
            chunk_id,
            vector_file,
            vector_id,
            doc_type,
            topic,
            created_at,
            user,
            source,
        ) = row

        metadata: dict[str, Any] = {
            "chunk_db_id": chunk_db_id,
            "source": source,
            "chunk_id": chunk_id,
            "start_char": start_char,
            "end_char": end_char,
            "length": length,
            "vector_file": vector_file,
            "vector_id": vector_id,
            "doc_type": self._normalize_doc_type(doc_type),
            "topic": topic,
            "created_at": created_at,
            "user": user,
        }

        return DocumentChunk(content=content, metadata=metadata, embedding=embedding)

    def _fetch_chunk_by_db_id(
        self,
        cursor: sqlite3.Cursor,
        chunk_db_id: int,
        *,
        load_embedding: bool = False,
    ) -> DocumentChunk | None:
        """Fetch a chunk by database row id.

        Returns:
            DocumentChunk if found; otherwise None.
        """
        cursor.execute(
            """
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
            WHERE c.id = ?
            """,
            (int(chunk_db_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        labels = self._load_labels(cursor, int(chunk_db_id))
        chunk = self._build_chunk_from_row(row)
        if labels:
            chunk.metadata["labels"] = labels

        if load_embedding and chunk.metadata.get("vector_file"):
            vector_path = Path(chunk.metadata["vector_file"])
            if vector_path.exists():
                chunk.embedding = self._load_embedding(vector_path)

        return chunk

    def _fetch_chunk_by_vector_id(
        self,
        cursor: sqlite3.Cursor,
        vector_id: int,
    ) -> DocumentChunk | None:
        """Fetch a chunk by FAISS/embedding vector id.

        Returns:
            DocumentChunk if found; otherwise None.
        """
        cursor.execute(
            "SELECT id FROM chunks WHERE vector_id = ?",
            (int(vector_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._fetch_chunk_by_db_id(cursor, int(row[0]))

    def _fetch_chunk_by_offset(
        self,
        cursor: sqlite3.Cursor,
        idx: int,
    ) -> DocumentChunk | None:
        """Fetch chunk by offset in insertion order.

        Returns:
            DocumentChunk if found; otherwise None.
        """
        cursor.execute(
            "SELECT id FROM chunks ORDER BY id LIMIT 1 OFFSET ?",
            (int(idx),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._fetch_chunk_by_db_id(cursor, int(row[0]))

    def _load_embedding(self, vector_path: Path) -> np.ndarray | None:
        """Placeholder for embedding loader implemented by subclasses.

        Returns:
            Loaded embedding array or None if unavailable.
        """
        raise NotImplementedError
