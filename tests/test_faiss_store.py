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
