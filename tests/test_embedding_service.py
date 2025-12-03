"""Tests for EmbeddingService class."""

import os
from unittest.mock import patch

import numpy as np
import pytest
from openai import APIConnectionError

from app import EmbeddingService
from app.config import config


def test_init_with_api_key(embedding_service_factory) -> None:
    service = embedding_service_factory(model="text-embedding-3-small")
    assert service.model == "text-embedding-3-small"
    assert service.client.api_key == "test-key"


def test_init_with_env_api_key() -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        service = EmbeddingService(model="text-embedding-3-small")
        assert service.model == "text-embedding-3-small"
        assert service.client.api_key == "env-key"


def test_init_default_model(embedding_service) -> None:
    service = embedding_service
    assert service.model == config.EMBEDDING_MODEL


def test_get_embedding_success(openai_embeddings_factory, embedding_service) -> None:
    mock_api = openai_embeddings_factory("single_success")
    service = embedding_service
    result = service.get_embedding("test text")

    mock_api.assert_called_once_with(
        model="text-embedding-3-small",
        input="test text",
    )

    assert isinstance(result, np.ndarray)
    expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    np.testing.assert_array_equal(result, np.array(expected_embedding))


def test_get_embedding_api_error(openai_embeddings_factory, embedding_service) -> None:
    openai_embeddings_factory("error", error_message="API Error")
    service = embedding_service

    with pytest.raises(Exception, match="API Error"):
        service.get_embedding("test text")


def test_get_embeddings_batch_success(
    openai_embeddings_factory, embedding_service
) -> None:
    mock_api = openai_embeddings_factory("batch_success")
    service = embedding_service
    texts = ["text1", "text2", "text3"]
    results = service.get_embeddings_batch(texts)

    mock_api.assert_called_once_with(
        model="text-embedding-3-small",
        input=texts,
    )

    assert len(results) == 3
    expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    for i, result in enumerate(results):
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(expected_embeddings[i]))


def test_get_embeddings_batch_with_batching(
    openai_embeddings_factory, embedding_service
) -> None:
    mock_api = openai_embeddings_factory("multiple_batches")
    service = embedding_service
    texts = ["text1", "text2", "text3", "text4"]
    results = service.get_embeddings_batch(texts, batch_size=2)

    assert mock_api.call_count == 2

    mock_api.assert_any_call(
        model="text-embedding-3-small",
        input=["text1", "text2"],
    )
    mock_api.assert_any_call(
        model="text-embedding-3-small",
        input=["text3", "text4"],
    )

    assert len(results) == 4
    expected_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    for i, result in enumerate(results):
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(expected_embeddings[i]))


def test_get_embeddings_batch_api_error(
    openai_embeddings_factory, embedding_service
) -> None:
    openai_embeddings_factory("error", error_message="Batch API Error")
    service = embedding_service
    texts = ["text1", "text2"]

    with pytest.raises(Exception, match="Batch API Error"):
        service.get_embeddings_batch(texts)


def test_get_embeddings_batch_empty_list(
    openai_embeddings_api_mock, embedding_service
) -> None:
    service = embedding_service
    results = service.get_embeddings_batch([])
    openai_embeddings_api_mock.assert_not_called()
    assert results == []


def test_get_embeddings_batch_partial_failure(
    openai_embeddings_factory, embedding_service
) -> None:
    mock_api = openai_embeddings_factory("partial_failure")
    service = embedding_service
    texts = ["text1", "text2", "text3"]

    with pytest.raises(Exception, match="Second batch failed"):
        service.get_embeddings_batch(texts, batch_size=1)

    assert mock_api.call_count == 2


# Integration tests that require a real OpenAI API key
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_real_api_single_embedding() -> None:
    """Test single embedding generation with real OpenAI API."""
    service = EmbeddingService(model="text-embedding-3-small")

    text = "This is a test sentence for embedding generation."
    try:
        embedding = service.get_embedding(text)
    except APIConnectionError as exc:  # pragma: no cover - network dependent
        pytest.skip(f"OpenAI not reachable: {exc!s}")
    else:
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 1536  # text-embedding-3-small dimension

        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01  # Allow small floating point errors


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_real_api_batch_embeddings() -> None:
    service = EmbeddingService(model="text-embedding-3-small")

    texts = [
        "This is the first test sentence.",
        "Here is another sentence for testing.",
        "A third sentence to complete the batch.",
    ]

    try:
        embeddings = service.get_embeddings_batch(texts, batch_size=2)
    except APIConnectionError as exc:  # pragma: no cover - network dependent
        pytest.skip(f"OpenAI not reachable: {exc!s}")
    else:
        assert len(embeddings) == 3

        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
            assert embedding.shape[0] == 1536

            norm = np.linalg.norm(embedding)
            assert 0.99 <= norm <= 1.01

        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_real_api_empty_text_handling() -> None:
    service = EmbeddingService(model="text-embedding-3-small")

    try:
        embedding_empty = service.get_embedding("")
        embedding_whitespace = service.get_embedding("   \n\t   ")
    except APIConnectionError as exc:  # pragma: no cover - network dependent
        pytest.skip(f"OpenAI not reachable: {exc!s}")
    else:
        assert isinstance(embedding_empty, np.ndarray)
        assert embedding_empty.shape[0] == 1536

        assert isinstance(embedding_whitespace, np.ndarray)
        assert embedding_whitespace.shape[0] == 1536


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
def test_real_api_different_text_lengths() -> None:
    service = EmbeddingService(model="text-embedding-3-small")

    long_text = (
        "This is a much longer text that contains multiple sentences "
        "and provides more context. "
    ) * 3

    texts = [
        "Short.",
        (
            "This is a much longer text that contains multiple sentences "
            "and provides more context. "
        ),
        (
            "This is a much longer text that contains multiple sentences "
            "and provides more context. "
        ),
        long_text,
    ]

    try:
        embeddings = service.get_embeddings_batch(texts)
    except APIConnectionError as exc:  # pragma: no cover - network dependent
        pytest.skip(f"OpenAI not reachable: {exc!s}")
    else:
        for embedding in embeddings:
            assert embedding.shape[0] == 1536

        assert not np.allclose(embeddings[0], embeddings[1])
        assert np.allclose(embeddings[1], embeddings[2])
        assert not np.allclose(embeddings[2], embeddings[3])
