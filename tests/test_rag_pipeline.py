"""Integration tests for RAG pipeline components."""

import json

import pytest

from app.core import DocumentLoader, TextChunker

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
MIN_VALID_SCORE = 0.0
MAX_VALID_SCORE = 1.0


def test_document_to_chunks_pipeline(sample_document_path):
    if not sample_document_path.exists():
        pytest.skip("Sample document not found")

    range_start_index = 1
    minimum_chunks = 0

    text = DocumentLoader.load_document(sample_document_path)
    chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    chunks = chunker.chunk_text(text, source=sample_document_path.name)

    assert len(chunks) > minimum_chunks
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(
        chunk.metadata["source"] == sample_document_path.name for chunk in chunks
    )

    for i in range(range_start_index, len(chunks)):
        prev_end = chunks[i - range_start_index].metadata["end_char"]
        curr_start = chunks[i].metadata["start_char"]
        assert prev_end - curr_start == DEFAULT_CHUNK_OVERLAP  # Expected overlap


def test_mock_rag_pipeline(
    sample_document_path, mock_embedding_service, temp_vector_store
) -> None:
    if not sample_document_path.exists():
        pytest.skip("Sample document not found")

    top_k_results = 3

    chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    text = DocumentLoader.load_document(sample_document_path)
    chunks = chunker.chunk_text(text, source=sample_document_path.name)
    test_query = "What is machine learning?"

    for chunk in chunks:
        chunk.embedding = mock_embedding_service.get_embedding(chunk.content)

    temp_vector_store.add_chunks(chunks)

    query_embedding = mock_embedding_service.get_embedding(test_query)
    results = temp_vector_store.search(query_embedding, top_k=top_k_results)

    assert len(results) == top_k_results, (
        f"Expected {top_k_results} results, got {len(results)}"
    )
    assert all(MIN_VALID_SCORE <= score <= MAX_VALID_SCORE for _, score in results), (
        "All scores should be within valid range [0.0, 1.0]"
    )

    for chunk, score in results:
        assert hasattr(chunk, "content"), "Chunk should have content attribute"
        assert hasattr(chunk, "metadata"), "Chunk should have metadata attribute"
        assert "source" in chunk.metadata, "Chunk metadata should contain source"
        assert isinstance(score, float), f"Score should be float, got {type(score)}"
        assert len(chunk.content.strip()) > 0, "Chunk content should not be empty"


def test_large_document_processing(
    mock_embedding_service, large_document_setup, temp_vector_store
):
    _, doc_path, _ = large_document_setup

    min_large_doc_chunks = 100
    large_doc_top_k_results = 10

    chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)

    text = DocumentLoader.load_document(doc_path)
    chunks = chunker.chunk_text(text, source="large_test")

    for chunk in chunks:
        chunk.embedding = mock_embedding_service.get_embedding(chunk.content)

    temp_vector_store.add_chunks(chunks)

    assert len(chunks) > min_large_doc_chunks  # Should create many chunks
    assert temp_vector_store.embeddings is not None
    assert temp_vector_store.embeddings.shape[0] == len(chunks)

    query_embedding = mock_embedding_service.get_embedding("artificial intelligence")
    results = temp_vector_store.search(query_embedding, top_k=large_doc_top_k_results)

    assert len(results) == large_doc_top_k_results
    assert all(
        isinstance(score, float) and MIN_VALID_SCORE <= score <= MAX_VALID_SCORE
        for _, score in results
    )


def test_evaluation_dataset_structure(evaluation_dataset_path):
    if not evaluation_dataset_path.exists():
        pytest.skip("Evaluation dataset not found")

    with evaluation_dataset_path.open(encoding="utf-8") as f:
        eval_data = json.load(f)

    assert "dataset_info" in eval_data
    assert "test_cases" in eval_data

    dataset_info = eval_data["dataset_info"]
    test_cases = eval_data["test_cases"]

    required_info_fields = ["name", "version", "total_questions"]
    for field in required_info_fields:
        assert field in dataset_info

    assert len(test_cases) == dataset_info["total_questions"]

    required_case_fields = ["id", "category", "question", "ideal_answer", "difficulty"]
    valid_categories = {
        "factual_retrieval",
        "conceptual_understanding",
        "multi_hop_reasoning",
        "comparison_questions",
        "contextual_questions",
    }
    valid_difficulties = {"easy", "medium", "hard"}

    for case in test_cases:
        for field in required_case_fields:
            assert field in case, (
                f"Missing field '{field}' in test case {case.get('id', 'unknown')}"
            )

        assert case["category"] in valid_categories
        assert case["difficulty"] in valid_difficulties
        assert isinstance(case["question"], str)
        assert len(case["question"]) > 0
        assert isinstance(case["ideal_answer"], str)
        assert len(case["ideal_answer"]) > 0


def test_evaluation_dataset_statistics(evaluation_dataset_path):
    if not evaluation_dataset_path.exists():
        pytest.skip("Evaluation dataset not found")

    min_categories_count = 3
    min_difficulties_count = 2
    category_dominance_threshold = 0.8

    with evaluation_dataset_path.open(encoding="utf-8") as f:
        eval_data = json.load(f)

    test_cases = eval_data["test_cases"]

    categories = {}
    difficulties = {}

    for case in test_cases:
        cat = case["category"]
        diff = case["difficulty"]
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

    assert len(categories) >= min_categories_count, (
        "Should have multiple question categories"
    )
    assert len(difficulties) >= min_difficulties_count, (
        "Should have multiple difficulty levels"
    )

    total_cases = len(test_cases)
    for count in categories.values():
        assert count <= total_cases * category_dominance_threshold, (
            "No single category should dominate"
        )

    for count in difficulties.values():
        assert count <= total_cases * category_dominance_threshold, (
            "No single difficulty should dominate"
        )
