"""Vector store adapters and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from app.config import config

from .faiss_store import FaissVectorStore
from .sqlite_store import SQLiteVectorStore

if TYPE_CHECKING:
    from pathlib import Path

VectorBackend = Literal["faiss", "sqlite"]


def get_vector_store(
    store: VectorBackend = "faiss",
    *,
    db_path: Path | None = None,
    vectors_dir: Path | None = None,
    index_path: Path | None = None,
    raw_top_k_multiplier: int | None = None,
) -> FaissVectorStore | SQLiteVectorStore:
    """Return a configured vector store instance.

    Raises:
        ValueError: If an unsupported backend is requested.
    """
    backend_value = store
    if db_path is None:
        db_path = config.VECTOR_STORE_DB_PATH
    backend = backend_value.lower()

    if backend == "faiss":
        return FaissVectorStore(
            db_path=db_path,
            index_path=(
                index_path if index_path is not None else config.FAISS_INDEX_PATH
            ),
            raw_top_k_multiplier=(
                raw_top_k_multiplier
                if raw_top_k_multiplier is not None
                else config.VECTOR_RAW_TOP_K_MULTIPLIER
            ),
        )

    if backend == "sqlite":
        return SQLiteVectorStore(
            db_path=db_path,
            vectors_dir=(
                vectors_dir if vectors_dir is not None else config.VECTOR_STORE_DIR
            ),
        )

    msg = f"Unsupported vector store backend: {store}"
    raise ValueError(msg)


__all__ = ["FaissVectorStore", "SQLiteVectorStore", "VectorBackend", "get_vector_store"]
