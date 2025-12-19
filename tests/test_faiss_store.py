"""Unit tests for FaissVectorStore."""

from app import DocumentChunk, FaissVectorStore


def test_faiss_add_and_search(temp_faiss_store, sample_embedded_chunks):
    store = temp_faiss_store
    store.add_chunks(sample_embedded_chunks)

    assert store.index is not None
    assert store.index.ntotal == len(sample_embedded_chunks)

    query_embedding = sample_embedded_chunks[0].embedding
    results = store.search(query_embedding, top_k=2)

    assert len(results) == 2
    chunk, score = results[0]
    assert isinstance(chunk, DocumentChunk)
    assert isinstance(score, float)
    assert "source" in chunk.metadata


def test_faiss_persistence_roundtrip(temp_faiss_store, sample_embedded_chunks):
    store = temp_faiss_store
    store.add_chunks(sample_embedded_chunks)
    store.save()

    reloaded_store = FaissVectorStore(
        db_path=store.db_path,
        index_path=store.index_path,
    )
    reloaded_store.load()

    assert reloaded_store.index is not None
    assert reloaded_store.index.ntotal == len(sample_embedded_chunks)

    query_embedding = sample_embedded_chunks[0].embedding
    results = reloaded_store.search(query_embedding, top_k=2)
    assert len(results) == 2


def test_faiss_metadata_and_labels_roundtrip(temp_faiss_store, mock_embeddings):
    store = temp_faiss_store
    content = "Persisted chunk with metadata"
    chunk = DocumentChunk(
        content=content,
        metadata={
            "source": "meta_doc.txt",
            "chunk_id": 7,
            "start_char": 0,
            "end_char": len(content),
            "length": len(content),
            "doc_type": "note",
            "topic": "ml",
            "user": "tester",
            "labels": ["alpha", "beta"],
        },
        embedding=mock_embeddings(content),
    )

    store.add_chunks([chunk])
    store.save()

    reloaded_store = FaissVectorStore(
        db_path=store.db_path,
        index_path=store.index_path,
        raw_top_k_multiplier=store.raw_top_k_multiplier,
    )
    reloaded_store.load()

    assert len(reloaded_store.chunks) == 1
    loaded_chunk = reloaded_store.chunks[0]
    assert loaded_chunk.metadata.get("doc_type") == "note"
    assert loaded_chunk.metadata.get("topic") == "ml"
    assert loaded_chunk.metadata.get("user") == "tester"
    assert loaded_chunk.metadata.get("labels") == ["alpha", "beta"]
    assert isinstance(loaded_chunk.metadata.get("vector_id"), int)

    results = reloaded_store.search(mock_embeddings(content), top_k=1)
    assert len(results) == 1
    result_chunk, _score = results[0]
    assert result_chunk.metadata.get("labels") == ["alpha", "beta"]


def test_faiss_search_respects_raw_top_k_limits(temp_faiss_store, mock_embeddings):
    store = temp_faiss_store
    chunk_one = DocumentChunk(
        content="Chunk one",
        metadata={"source": "raw_topk.txt", "chunk_id": 0},
        embedding=mock_embeddings("Chunk one"),
    )
    chunk_two = DocumentChunk(
        content="Chunk two",
        metadata={"source": "raw_topk.txt", "chunk_id": 1},
        embedding=mock_embeddings("Chunk two"),
    )

    store.add_chunks([chunk_one, chunk_two])
    results = store.search(mock_embeddings("Chunk one"), top_k=5)

    assert len(results) == 2  # ntotal < requested top_k should cap at ntotal
    top_chunk, _score = results[0]
    assert top_chunk.content == "Chunk one"
