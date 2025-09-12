"""Test configuration and fixtures for LCEngine tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.core import DocumentChunk, SQLiteVectorStore


class MockEmbeddingService:
    """Mock embedding service for testing without API calls."""

    def __init__(self, dimension: int = 384) -> None:
        """Initialize mock embedding service.

        Args:
            dimension: Dimensionality of generated embeddings.
        """
        self.dimension = dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding."""
        hash_val = hash(text.lower()) % (2**31)
        rng = np.random.default_rng(hash_val)
        embedding = rng.normal(0, 1, self.dimension)
        return (embedding / np.linalg.norm(embedding)).astype(np.float32)

    def get_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[np.ndarray]:
        """Generate batch of mock embeddings."""
        return [self.get_embedding(text) for text in texts]


def create_mock_openai_response(embeddings: list[list[float]]) -> Mock:
    """Create a mock OpenAI API response.

    Args:
        embeddings: List of embedding vectors to return.

    Returns:
        Mock object representing OpenAI API response.
    """
    mock_response = Mock()
    mock_response.data = [Mock(embedding=emb) for emb in embeddings]
    return mock_response


@pytest.fixture
def mock_openai_create():
    """Base fixture that patches OpenAI embeddings.create method.

    This is the foundation fixture that others can build upon.
    Returns the mock object directly without any pre-configuration.
    """
    with patch("openai.resources.embeddings.Embeddings.create") as mock_create:
        yield mock_create


@pytest.fixture
def mock_single_embedding(mock_openai_create):
    """Fixture for single embedding API calls with standard test data."""
    mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_response = create_mock_openai_response([mock_embedding])
    mock_openai_create.return_value = mock_response
    return mock_openai_create


@pytest.fixture
def mock_batch_embeddings(mock_openai_create):
    """Fixture for batch embedding API calls with standard test data."""
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    mock_response = create_mock_openai_response(mock_embeddings)
    mock_openai_create.return_value = mock_response
    return mock_openai_create


@pytest.fixture
def mock_api_error(mock_openai_create):
    """Fixture that simulates API errors."""
    mock_openai_create.side_effect = Exception("API Error")
    return mock_openai_create


@pytest.fixture
def mock_batch_api_error(mock_openai_create):
    """Fixture that simulates batch API errors."""
    mock_openai_create.side_effect = Exception("Batch API Error")
    return mock_openai_create


@pytest.fixture
def mock_multiple_batches(mock_openai_create):
    """Fixture for testing multiple batch API calls."""
    mock_embedding1 = [[0.1, 0.2], [0.3, 0.4]]
    mock_embedding2 = [[0.5, 0.6], [0.7, 0.8]]

    mock_response1 = create_mock_openai_response(mock_embedding1)
    mock_response2 = create_mock_openai_response(mock_embedding2)

    mock_openai_create.side_effect = [mock_response1, mock_response2]
    return mock_openai_create


@pytest.fixture
def mock_partial_failure(mock_openai_create):
    """Fixture for testing partial batch failures."""
    mock_response = create_mock_openai_response([[0.1, 0.2]])
    mock_openai_create.side_effect = [mock_response, Exception("Second batch failed")]
    return mock_openai_create


@pytest.fixture
def mock_embeddings():
    """Fixture to create mock embeddings."""

    def _create_mock_embedding(text: str, dimension: int = 384) -> np.ndarray:
        service = MockEmbeddingService(dimension)
        return service.get_embedding(text)

    return _create_mock_embedding


@pytest.fixture
def mock_embedding_service():
    """Fixture to create mock embedding service."""
    return MockEmbeddingService()


@pytest.fixture
def temp_vector_store() -> Generator[SQLiteVectorStore, None, None]:
    """Create temporary SQLite vector store for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        db_path = temp_path / "test_store.db"
        vectors_dir = temp_path / "vectors"

        store = SQLiteVectorStore(db_path, vectors_dir)
        yield store


@pytest.fixture
def sample_chunks(mock_embeddings):
    """Create sample document chunks for testing."""
    chunks = []
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are computational models inspired by the brain.",
        "Deep learning uses multiple layers to learn complex patterns.",
        "Supervised learning uses labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
    ]

    for i, text in enumerate(texts):
        chunk = DocumentChunk(
            content=text,
            metadata={
                "source": f"test_doc_{i // 3}.txt",
                "chunk_id": i,
                "start_char": i * 100,
                "end_char": (i + 1) * 100,
                "length": len(text),
            },
            embedding=mock_embeddings(text),
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def sample_document_path():
    """Path to the sample ML document.

    This fixture uses persistent files for: hand-crafted data,
    complex content, read-only test data shared across tests.
    """
    return Path("data/sample_ml_document.txt")


@pytest.fixture
def evaluation_dataset_path():
    """Path to the evaluation dataset."""
    return Path("data/evaluation_dataset.json")


@pytest.fixture
def large_document_setup():
    """Create a large document and temporary directory for testing.

    This fixture creates temporary files for each test to ensure isolation.
    Use this pattern for: generated data, modified files, output files.
    """
    large_text = (
        "Machine learning is a subset of artificial intelligence. "
        * 1000  # LARGE_DOC_REPEAT_COUNT
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        doc_path = temp_path / "large_doc.txt"
        doc_path.write_text(large_text)

        yield temp_path, doc_path, large_text
