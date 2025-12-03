"""Unit tests for SQLiteVectorStore."""

import sqlite3

import numpy as np

from app import DocumentChunk, SQLiteVectorStore


def test_store_initialization(temp_vector_store):
    store = temp_vector_store

    db_path = store.db_path
    vectors_dir = store.vectors_dir

    assert db_path.exists()
    assert vectors_dir.exists()
    assert store.chunks == []
    assert store.embeddings is None


def test_metadata_and_labels_persistence(temp_vector_store, mock_embeddings):
    store = temp_vector_store
    chunk = DocumentChunk(
        content="Test chunk with metadata",
        metadata={
            "source": "test_doc.txt",
            "chunk_id": 0,
            "start_char": 0,
            "end_char": 10,
            "length": 10,
            "doc_type": "note",
            "topic": "ml",
            "user": "tester",
            "labels": ["ml", "exam"],
        },
        embedding=mock_embeddings("Test chunk with metadata"),
    )

    store.add_chunks([chunk])

    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, doc_type, topic, user, vector_id FROM chunks LIMIT 1",
        )
        chunk_id, doc_type, topic, user, vector_id = cursor.fetchone()
        cursor.execute("SELECT label FROM labels WHERE chunk_id = ?", (chunk_id,))
        labels = sorted(label for (label,) in cursor.fetchall())

    assert doc_type == "note"
    assert topic == "ml"
    assert user == "tester"
    assert vector_id is not None
    assert labels == ["exam", "ml"]


def test_labels_loaded_with_chunks(temp_vector_store, mock_embeddings):
    store = temp_vector_store
    chunk = DocumentChunk(
        content="Persisted chunk",
        metadata={
            "source": "persisted.txt",
            "chunk_id": 1,
            "start_char": 0,
            "end_char": 20,
            "length": 20,
            "doc_type": "mistake",
            "labels": ["review"],
        },
        embedding=mock_embeddings("Persisted chunk"),
    )

    store.add_chunks([chunk])
    store.save()

    new_store = SQLiteVectorStore(store.db_path, store.vectors_dir)
    new_store.load()

    assert len(new_store.chunks) == 1
    loaded_chunk = new_store.chunks[0]
    assert loaded_chunk.metadata.get("doc_type") == "mistake"
    assert loaded_chunk.metadata.get("labels") == ["review"]


def test_add_chunks(temp_vector_store, sample_embedded_chunks):
    store = temp_vector_store

    store.add_chunks(sample_embedded_chunks)

    assert len(store.chunks) == len(sample_embedded_chunks)
    assert store.embeddings is not None
    assert store.embeddings.shape[0] == len(sample_embedded_chunks)

    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        assert count == len(sample_embedded_chunks)


def test_search_functionality(temp_vector_store, sample_embedded_chunks):
    store = temp_vector_store
    store.add_chunks(sample_embedded_chunks)

    query_embedding = sample_embedded_chunks[0].embedding
    results = store.search(query_embedding, top_k=3)

    assert len(results) == 3
    assert all(len(result) == 2 for result in results)  # (chunk, score) tuples

    chunk, score = results[0]
    assert isinstance(chunk, DocumentChunk)
    assert isinstance(score, float)
    assert score > 0.9  # High similarity expected


def test_empty_search(temp_vector_store, mock_embeddings):
    store = temp_vector_store

    query_embedding = mock_embeddings("test query")
    results = store.search(query_embedding, top_k=5)

    assert len(results) == 0


def test_persistence(temp_vector_store, sample_embedded_chunks):
    store = temp_vector_store
    store.add_chunks(sample_embedded_chunks)
    store.save()

    new_store = SQLiteVectorStore(store.db_path, store.vectors_dir)
    new_store.load()

    assert len(new_store.chunks) == len(sample_embedded_chunks)
    assert new_store.embeddings is not None
    assert new_store.embeddings.shape[0] == len(sample_embedded_chunks)

    # Test search consistency
    query_embedding = sample_embedded_chunks[0].embedding
    original_results = store.search(query_embedding, top_k=3)
    loaded_results = new_store.search(query_embedding, top_k=3)

    assert len(original_results) == len(loaded_results)

    for (_, score1), (_, score2) in zip(original_results, loaded_results, strict=False):
        assert abs(score1 - score2) < 1e-6


def test_cosine_similarity(temp_vector_store):
    store = temp_vector_store

    vec1 = np.array([1, 0, 0], dtype=np.float32)
    vec2 = np.array([0, 1, 0], dtype=np.float32)
    vec3 = np.array([1, 1, 0], dtype=np.float32) / np.sqrt(2)  # 45 degrees

    embeddings = np.vstack([vec1, vec2, vec3])

    similarities = store.cosine_similarity(vec1, embeddings)

    assert abs(similarities[0] - 1.0) < 1e-6  # Same vector
    assert abs(similarities[1] - 0.0) < 1e-6  # Orthogonal
    assert abs(similarities[2] - (1 / np.sqrt(2))) < 1e-6  # 45 degrees


def test_add_chunks_without_embeddings(temp_vector_store, sample_text_chunks):
    store = temp_vector_store

    # Use text chunks which have no embeddings
    store.add_chunks(sample_text_chunks)

    # Should have no chunks added (all skipped)
    assert len(store.chunks) == 0
    assert store.embeddings is None
