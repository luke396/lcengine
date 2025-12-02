"""Test configuration and fixtures for LCEngine tests.

This module provides reusable test fixtures organized by functionality:
- Constants and test data
- Mock services and API responses
- EmbeddingService fixtures
- Text processing fixtures
- Vector store fixtures
- Sample data factories
- Integration test helpers
"""

import hashlib
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, create_autospec, patch

import numpy as np
import pytest

from app import (
    ConversationManager,
    DocumentChunk,
    EmbeddingService,
    RAGPipeline,
    SQLiteVectorStore,
    TextChunker,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "data"


class TestConstants:
    """Centralized test constants to avoid repetition across test files.

    All test constants are defined here to maintain consistency across
    the test suite and make it easy to update values globally.
    """

    # API Configuration
    TEST_API_KEY = "test-key"
    TEST_OPENAI_MODEL = "text-embedding-3-small"
    DEFAULT_EMBEDDING_DIMENSION = 384

    # Text Chunking Configuration
    SMALL_CHUNK_SIZE = 100
    SMALL_CHUNK_OVERLAP = 20
    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 100
    LARGE_CHUNK_SIZE = 10000
    LARGE_CHUNK_OVERLAP = 1000


class MockEmbeddingService:
    """Mock embedding service for testing without API calls.

    Generates deterministic embeddings based on text content hash,
    ensuring consistent test results across runs.
    """

    def __init__(
        self, dimension: int = TestConstants.DEFAULT_EMBEDDING_DIMENSION
    ) -> None:
        """Initialize mock embedding service.

        Args:
            dimension: Dimensionality of generated embeddings.
        """
        self.dimension = dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding based on text hash."""
        seed = int.from_bytes(
            hashlib.sha256(text.lower().encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        rng = np.random.default_rng(seed)
        embedding = rng.normal(0, 1, self.dimension)
        return (embedding / np.linalg.norm(embedding)).astype(np.float32)

    def get_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[np.ndarray]:
        """Generate batch of mock embeddings."""
        return [self.get_embedding(text) for text in texts]


def create_mock_openai_response(embeddings: list[list[float]]) -> Mock:
    """Create a mock OpenAI embeddings API response.

    Args:
        embeddings: List of embedding vectors to return.

    Returns:
        Mock object representing OpenAI embeddings API response.
    """
    mock_response = Mock()
    mock_response.data = [Mock(embedding=emb) for emb in embeddings]
    return mock_response


def create_mock_chat_response(content: str | None) -> Mock:
    """Create a mock OpenAI chat completion response.

    Args:
        content: The content for the chat completion response.

    Returns:
        Mock object representing OpenAI chat completion response.
    """
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=content))]
    return mock_response


@pytest.fixture
def openai_embeddings_api_mock():
    """Base fixture that patches OpenAI embeddings.create method.

    This is the foundation fixture that others can build upon.
    Returns the mock object directly without any pre-configuration.
    """
    with patch("openai.resources.embeddings.Embeddings.create") as mock_create:
        yield mock_create


@pytest.fixture
def openai_embeddings_factory(openai_embeddings_api_mock):
    """Factory for creating OpenAI embeddings API mocks with different scenarios."""

    def _create_mock(  # noqa: ANN202
        scenario="single_success",
        embeddings=None,
        error_message="API Error",
        side_effects=None,
    ):
        """Create a mock based on scenario type.

        Args:
            scenario: Type of mock ('single_success', 'batch_success', 'error',
                'multiple_batches', 'partial_failure')
            embeddings: Custom embeddings to return, or None for defaults
            error_message: Custom error message for error scenarios
            side_effects: Custom side effects list for complex scenarios
        """
        openai_embeddings_api_mock.reset_mock()
        openai_embeddings_api_mock.side_effect = None
        openai_embeddings_api_mock.return_value = None

        if scenario == "single_success":
            mock_embedding = embeddings or [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_response = create_mock_openai_response([mock_embedding])
            openai_embeddings_api_mock.return_value = mock_response
        elif scenario == "batch_success":
            mock_embeddings = embeddings or [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
            mock_response = create_mock_openai_response(mock_embeddings)
            openai_embeddings_api_mock.return_value = mock_response
        elif scenario == "error":
            openai_embeddings_api_mock.side_effect = Exception(error_message)
        elif scenario == "multiple_batches":
            if side_effects:
                openai_embeddings_api_mock.side_effect = side_effects
            else:
                mock_embedding1 = [[0.1, 0.2], [0.3, 0.4]]
                mock_embedding2 = [[0.5, 0.6], [0.7, 0.8]]
                mock_response1 = create_mock_openai_response(mock_embedding1)
                mock_response2 = create_mock_openai_response(mock_embedding2)
                openai_embeddings_api_mock.side_effect = [
                    mock_response1,
                    mock_response2,
                ]
        elif scenario == "partial_failure":
            mock_response = create_mock_openai_response([[0.1, 0.2]])
            openai_embeddings_api_mock.side_effect = [
                mock_response,
                Exception("Second batch failed"),
            ]

        return openai_embeddings_api_mock

    return _create_mock


@pytest.fixture
def embedding_service_factory():
    """Factory for creating EmbeddingService instances with different configurations."""

    def _create_service(api_key=None, model=None):  # noqa: ANN202
        """Create an EmbeddingService instance.

        Args:
            api_key: API key to use, defaults to TestConstants.TEST_API_KEY
            model: Model to use, defaults to
                TestConstants.TEST_OPENAI_MODEL or config default
        """
        api_key = api_key or TestConstants.TEST_API_KEY

        if model is not None:
            return EmbeddingService(api_key=api_key, model=model)
        return EmbeddingService(api_key=api_key)

    return _create_service


@pytest.fixture
def embedding_service(embedding_service_factory):
    """Default EmbeddingService with test API key for most tests."""
    return embedding_service_factory()


@pytest.fixture
def text_chunker_factory():
    """Factory fixture that creates ``TextChunker`` instances on demand."""
    presets: dict[str, tuple[int, int]] = {
        "small": (
            TestConstants.SMALL_CHUNK_SIZE,
            TestConstants.SMALL_CHUNK_OVERLAP,
        ),
        "default": (
            TestConstants.DEFAULT_CHUNK_SIZE,
            TestConstants.DEFAULT_CHUNK_OVERLAP,
        ),
        "large": (
            TestConstants.LARGE_CHUNK_SIZE,
            TestConstants.LARGE_CHUNK_OVERLAP,
        ),
    }

    def _create_chunker(
        name: str = "default",
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> TextChunker:
        if chunk_size is None or overlap is None:
            try:
                preset_chunk_size, preset_overlap = presets[name]
            except KeyError as exc:
                msg = f"Unknown text chunker preset: {name}"
                raise ValueError(msg) from exc
            chunk_size = preset_chunk_size if chunk_size is None else chunk_size
            overlap = preset_overlap if overlap is None else overlap

        return TextChunker(chunk_size=chunk_size, overlap=overlap)

    return _create_chunker


@pytest.fixture
def text_chunker_small(text_chunker_factory):
    """Text chunker configured for small chunks (100/20)."""
    return text_chunker_factory("small")


@pytest.fixture
def text_chunker_default(text_chunker_factory):
    """Text chunker configured with default settings (500/100)."""
    return text_chunker_factory("default")


@pytest.fixture
def text_chunker_large(text_chunker_factory):
    """Text chunker configured for large chunks (10000/1000)."""
    return text_chunker_factory("large")


@pytest.fixture(scope="session")
def mock_embedding_service():
    """Pre-configured MockEmbeddingService for consistent test embeddings."""
    return MockEmbeddingService()


@pytest.fixture
def mock_embeddings(mock_embedding_service):
    """Factory function to create mock embeddings using the service."""

    def _create_mock_embedding(
        text: str, dimension: int = TestConstants.DEFAULT_EMBEDDING_DIMENSION
    ) -> np.ndarray:
        if dimension != TestConstants.DEFAULT_EMBEDDING_DIMENSION:
            service = MockEmbeddingService(dimension)
            return service.get_embedding(text)
        return mock_embedding_service.get_embedding(text)

    return _create_mock_embedding


@pytest.fixture
def mock_embeddings_batch(mock_embedding_service):
    """Factory function to create batch mock embeddings using the service."""

    def _create_mock_embeddings_batch(
        texts: list[str], dimension: int = TestConstants.DEFAULT_EMBEDDING_DIMENSION
    ) -> list[np.ndarray]:
        if dimension != TestConstants.DEFAULT_EMBEDDING_DIMENSION:
            service = MockEmbeddingService(dimension)
            return service.get_embeddings_batch(texts)
        return mock_embedding_service.get_embeddings_batch(texts)

    return _create_mock_embeddings_batch


@pytest.fixture
def temp_vector_store(tmp_path) -> SQLiteVectorStore:
    """Create temporary SQLite vector store for testing."""
    db_path = tmp_path / "test_store.db"
    vectors_dir = tmp_path / "vectors"
    return SQLiteVectorStore(db_path, vectors_dir)


@pytest.fixture
def sample_text_chunks():
    """Create sample document chunks with text and metadata only (no embeddings)."""
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
            embedding=None,  # No embedding for text-only chunks
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def sample_embedded_chunks(sample_text_chunks, mock_embeddings):
    """Create sample document chunks with embeddings based on text chunks."""
    chunks = []
    for chunk in sample_text_chunks:
        embedded_chunk = DocumentChunk(
            content=chunk.content,
            metadata=chunk.metadata,
            embedding=mock_embeddings(chunk.content),
        )
        chunks.append(embedded_chunk)

    return chunks


@pytest.fixture(scope="session")
def sample_document_path():
    """Path to the sample ML document.

    This fixture uses persistent files for: hand-crafted data,
    complex content, read-only test data shared across tests.
    """
    return TEST_DATA_DIR / "sample_ml_document.txt"


@pytest.fixture(scope="session")
def evaluation_dataset_path():
    """Path to the evaluation dataset."""
    return TEST_DATA_DIR / "evaluation_dataset.json"


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


@pytest.fixture
def mock_rag_pipeline_empty_return():
    mock_pipeline = create_autospec(RAGPipeline, instance=True)
    mock_pipeline.query.return_value = []
    return mock_pipeline


@pytest.fixture
def conversation_manager(mock_rag_pipeline_empty_return):
    """Pre-configured ConversationManager with mock RAG pipeline."""
    manager = ConversationManager(
        rag_pipeline=mock_rag_pipeline_empty_return,
        openai_api_key=TestConstants.TEST_API_KEY,
    )
    with patch.object(
        manager.client.chat.completions,
        "create",
        return_value=create_mock_chat_response("Test response"),
    ):
        yield manager


@pytest.fixture
def performance_chunks_factory(mock_embedding_service):
    """Factory to create large batches of DocumentChunk for performance testing."""

    def _create_chunks(
        count: int, content_prefix: str = "Performance test chunk"
    ) -> list[DocumentChunk]:
        chunks = []
        for i in range(count):
            content = f"{content_prefix} {i}: " + "test content " * 10
            chunk = DocumentChunk(
                content=content,
                metadata={
                    "source": f"perf_doc_{i // 100}.txt",
                    "chunk_id": i,
                    "start_char": i * 100,
                    "end_char": (i + 1) * 100,
                    "length": len(content),
                },
                embedding=mock_embedding_service.get_embedding(content),
            )
            chunks.append(chunk)
        return chunks

    return _create_chunks


@pytest.fixture
def sample_chunks():
    """Create sample document chunks with scores for testing."""
    return [
        (
            DocumentChunk(
                content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "ml_doc.pdf", "chunk_id": 1},
            ),
            0.8,
        ),
        (
            DocumentChunk(
                content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "dl_doc.pdf", "chunk_id": 2},
            ),
            0.7,
        ),
        (
            DocumentChunk(
                content=(
                    "Natural language processing enables machines to understand text."
                ),
                metadata={"source": "nlp_doc.pdf", "chunk_id": 3},
            ),
            0.6,
        ),
    ]


@pytest.fixture
def rag_pipeline_factory(tmp_path):
    """Factory for creating RAGPipeline instances with patched API key."""

    def _create_pipeline(
        db_name: str = "test_store.db",
        chunk_size: int = 200,
        overlap: int = 50,
    ) -> RAGPipeline:
        with patch(
            "app.embeddings.config.get_openai_api_key",
            return_value=TestConstants.TEST_API_KEY,
        ):
            return RAGPipeline(
                openai_api_key=TestConstants.TEST_API_KEY,
                sqlite_db_path=tmp_path / db_name,
                vectors_dir=tmp_path / "vectors",
                chunk_size=chunk_size,
                overlap=overlap,
            )

    return _create_pipeline


@pytest.fixture
def conversation_manager_factory():
    """Factory fixture for creating ConversationManager instances."""

    def _create_conversation_manager(
        rag_pipeline, api_key: str = TestConstants.TEST_API_KEY
    ) -> ConversationManager:
        return ConversationManager(rag_pipeline=rag_pipeline, openai_api_key=api_key)

    return _create_conversation_manager


@pytest.fixture
def conversation_manager_chat_mock_factory():
    """Factory mock fixture for ConversationManager's client.chat.completions.create."""

    @contextmanager
    def _mock_conversation_manager_chat(  # noqa: ANN202
        conversation_manager, content: str | None = "Test response", side_effect=None
    ):
        with patch.object(
            conversation_manager.client.chat.completions,
            "create",
        ) as mock_create:
            if side_effect is not None:
                mock_create.side_effect = side_effect
                mock_create.return_value = None
            else:
                mock_create.side_effect = None
                mock_create.return_value = create_mock_chat_response(content)
            yield mock_create

    return _mock_conversation_manager_chat
