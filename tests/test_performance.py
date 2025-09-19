"""Performance and stress tests for LCEngine components.

The time and speed benchmarks in the tests are temporarily
determined by the performance of my local machine.
"""

import os
import time

import psutil

from app import DocumentChunk, TextChunker


def test_text_chunking_performance(text_chunker_default):
    large_text = "Machine learning is transforming industries. " * 5000

    chunker = text_chunker_default

    start_time = time.time()
    chunks = chunker.chunk_text(large_text, "performance_test")
    chunk_time = time.time() - start_time

    assert len(chunks) > 500, "Should create many chunks from large document"
    assert chunk_time < 0.010, f"Chunking took too long: {chunk_time:.3f}s"

    chars_per_second = len(large_text) / chunk_time
    chunks_per_second = len(chunks) / chunk_time

    assert chars_per_second > 10000000, (
        f"Character processing too slow: {chars_per_second:.0f} chars/s"
    )
    assert chunks_per_second > 100000, (
        f"Chunk creation too slow: {chunks_per_second:.1f} chunks/s"
    )


def test_vector_store_performance(
    temp_vector_store, performance_chunks_factory, mock_embedding_service
):
    store = temp_vector_store
    mock_service = mock_embedding_service

    chunks = performance_chunks_factory(1000, "Performance test content")

    storage_start = time.time()
    store.add_chunks(chunks)
    storage_time = time.time() - storage_start

    query_embedding = mock_service.get_embedding("test query")

    search_start = time.time()
    results = store.search(query_embedding, top_k=10)
    search_time = time.time() - search_start

    assert len(results) == 10, "Should return requested number of results"
    assert storage_time < 0.3, f"Storage too slow: {storage_time:.3f}s"
    assert search_time < 0.015, f"Search too slow: {search_time:.3f}s"


def test_memory_usage_stability(temp_vector_store, performance_chunks_factory):
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    store = temp_vector_store

    for round_num in range(5):
        chunks = performance_chunks_factory(200, f"Memory test chunk round {round_num}")

        store.add_chunks(chunks)

        current_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = current_memory - initial_memory

        assert memory_growth < 20, f"Excessive memory growth: {memory_growth:.1f}MB"


def test_very_large_chunks(text_chunker_large):
    chunker = text_chunker_large

    large_content = "This is a stress test with very large chunks. " * 25000

    start_time = time.time()
    chunks = chunker.chunk_text(large_content, "stress_test")
    processing_time = time.time() - start_time

    assert len(chunks) > 0, "Should create chunks from large content"
    assert processing_time < 0.005, f"Processing took too long: {processing_time:.3f}s"

    for chunk in chunks:
        assert len(chunk.content) <= 10000, "Chunk too large"
        assert len(chunk.content) > 0, "Empty chunk created"

    # Verify the configured overlap is respected between consecutive chunks
    if len(chunks) > 1:
        expected_overlap = 1000
        for i in range(len(chunks) - 1):
            prev = chunks[i]
            nxt = chunks[i + 1]

            prev_end = int(prev.metadata["end_char"])
            next_start = int(nxt.metadata["start_char"])

            assert prev_end - next_start == expected_overlap, (
                f"Incorrect overlap between chunk {i} and {i + 1}: "
                f"expected {expected_overlap}, got {prev_end - next_start}"
            )

            overlapped_prev = large_content[prev_end - expected_overlap : prev_end]
            overlapped_next = large_content[next_start : next_start + expected_overlap]
            assert overlapped_prev == overlapped_next, (
                f"Overlapped text mismatch at chunks {i} and {i + 1}"
            )


def test_many_small_chunks():
    chunker = TextChunker(chunk_size=50, overlap=10)

    content = ". ".join([f"Sentence {i}" for i in range(5000)])

    start_time = time.time()
    chunks = chunker.chunk_text(content, "many_chunks_test")
    processing_time = time.time() - start_time

    assert len(chunks) > 1000, f"Expected many chunks, got {len(chunks)}"
    assert processing_time < 0.005, f"Processing took too long: {processing_time:.3f}s"


def test_concurrent_operations_simulation(temp_vector_store, mock_embedding_service):
    store = temp_vector_store
    mock_service = mock_embedding_service

    total_operations = 100

    start_time = time.time()
    for i in range(total_operations):
        batch = [
            DocumentChunk(
                content=f"Concurrent test chunk {i}-{j}",
                metadata={
                    "source": f"concurrent_{i}.txt",
                    "chunk_id": j,
                    "start_char": j * 50,
                    "end_char": (j + 1) * 50,
                    "length": 25,
                },
                embedding=mock_service.get_embedding(f"content {i}-{j}"),
            )
            for j in range(2)
        ]

        store.add_chunks(batch)

        query_embedding = mock_service.get_embedding(f"query {i}")
        results = store.search(query_embedding, top_k=3)
        assert len(results) <= 3, "Search returned too many results"

    total_time = time.time() - start_time
    ops_per_second = total_operations / total_time

    assert ops_per_second > 2, f"Operations too slow: {ops_per_second:.1f} ops/s"
