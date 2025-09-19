"""Integration tests for LCEngine RAG pipeline end-to-end workflows."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest


def assert_valid_rag_result(result: list, min_chunks: int = 1) -> None:
    """Helper function to validate RAG query results."""
    assert result is not None, "Query should return results"
    assert isinstance(result, list), "Result should be a list of chunks with scores"
    assert len(result) >= min_chunks, (
        f"Query should return at least {min_chunks} relevant chunk(s)"
    )

    for chunk_with_score in result:
        assert len(chunk_with_score) == 2, (
            "Each result should be a tuple of (chunk, score)"
        )
        chunk, score = chunk_with_score
        assert hasattr(chunk, "content"), "Chunk should have content attribute"
        assert hasattr(chunk, "metadata"), "Chunk should have metadata attribute"
        assert -1.0 <= score <= 1.0, (
            f"Cosine similarity should be between -1.0 and 1.0, got {score}"
        )
        assert len(chunk.content.strip()) > 0, "Chunk content should not be empty"
        assert "source" in chunk.metadata, "Chunk metadata should contain source"


def assert_valid_conversation_result(result: dict) -> None:
    """Helper function to validate conversation manager results."""
    assert result is not None, "Conversation manager should return a response"
    assert isinstance(result, dict), "Response should be a dictionary"
    assert "answer" in result, "Response should contain answer key"
    assert len(result["answer"].strip()) > 0, "Answer should not be empty"
    assert "retrieved_contexts" in result, "Should contain retrieved contexts"


def create_test_documents(tmp_path: Path, documents: dict[str, str]) -> dict[str, Path]:
    """Helper function to create multiple test documents."""
    doc_paths = {}
    for filename, content in documents.items():
        doc_path = tmp_path / filename
        doc_path.write_text(content)
        doc_paths[filename] = doc_path
    return doc_paths


def mock_pipeline_processing(
    pipeline,
    sample_chunks: list,
    mock_embeddings,
    mock_embeddings_batch,
    query: str = "test query",
) -> ExitStack:
    """Context manager for mocking complete pipeline processing chain."""
    stack = ExitStack()
    stack.enter_context(
        patch.object(pipeline.chunker, "chunk_text", return_value=sample_chunks)
    )
    stack.enter_context(
        patch.object(
            pipeline.embedding_service,
            "get_embeddings_batch",
            return_value=mock_embeddings_batch([
                chunk.content for chunk in sample_chunks
            ]),
        )
    )
    stack.enter_context(
        patch.object(
            pipeline.embedding_service,
            "get_embedding",
            return_value=mock_embeddings(query),
        )
    )
    return stack


def test_complete_rag_workflow_text_document(  # noqa: PLR0913, PLR0917
    rag_pipeline_factory,
    conversation_manager_factory,
    sample_document_path,
    sample_text_chunks,
    mock_embeddings,
    mock_embeddings_batch,
    conversation_manager_chat_mock_factory,
):
    """Test complete workflow: document processing → storage → query → conversation."""
    query = "What is machine learning?"
    pipeline = rag_pipeline_factory()

    with mock_pipeline_processing(
        pipeline, sample_text_chunks, mock_embeddings, mock_embeddings_batch, query
    ):
        pipeline.process_document(sample_document_path)
        conversation_manager = conversation_manager_factory(pipeline)

        result = pipeline.query(query)
        assert_valid_rag_result(result)

        test_response_content = (
            "Machine learning is a subset of AI that enables computers to learn "
            "without being explicitly programmed."
        )
        with conversation_manager_chat_mock_factory(
            conversation_manager, content=test_response_content
        ) as mock_chat:
            conversation_result = conversation_manager.answer_question(query)
            assert_valid_conversation_result(conversation_result)

            mock_chat.assert_called_once()
            call_args = mock_chat.call_args
            assert "messages" in call_args.kwargs, "Should pass messages to OpenAI API"

        vector_store = pipeline.vector_store
        assert vector_store.embeddings is not None, "Embeddings matrix should be built"
        assert vector_store.embeddings.shape[0] > 0, "Should have processed chunks"
        assert vector_store.vectors_dir.exists(), "Vectors directory should exist"


def test_rag_pipeline_with_empty_document(
    rag_pipeline_factory, tmp_path, mock_embeddings
):
    """Test RAG pipeline behavior with empty document input."""
    pipeline = rag_pipeline_factory("test_empty.db")

    # Create empty document
    docs = create_test_documents(tmp_path, {"empty_doc.txt": ""})

    with (
        patch.object(pipeline.chunker, "chunk_text", return_value=[]),
        patch.object(
            pipeline.embedding_service,
            "get_embedding",
            return_value=mock_embeddings("test"),
        ),
    ):
        pipeline.process_document(docs["empty_doc.txt"])
        result = pipeline.query("What is this about?")
        assert result == [], "Empty document should return no results"


def test_conversation_manager_with_api_error(  # noqa: PLR0913, PLR0917
    rag_pipeline_factory,
    conversation_manager_factory,
    tmp_path,
    mock_embeddings,
    sample_text_chunks,
    mock_embeddings_batch,
):
    """Test conversation manager handles API errors gracefully."""
    pipeline = rag_pipeline_factory("test_api_error.db")

    # Setup successful document processing
    with mock_pipeline_processing(
        pipeline, sample_text_chunks, mock_embeddings, mock_embeddings_batch
    ):
        test_doc = tmp_path / "test_doc.txt"
        test_doc.write_text("Test content for API error handling")
        pipeline.process_document(test_doc)

    conversation_manager = conversation_manager_factory(pipeline)

    # Mock API to raise an error during conversation generation
    with (
        patch.object(
            conversation_manager.client.chat.completions, "create"
        ) as mock_chat,
        patch.object(
            conversation_manager.rag_pipeline.embedding_service, "get_embedding"
        ) as mock_query_embedding,
    ):
        mock_chat.side_effect = Exception("OpenAI API Error")
        mock_query_embedding.return_value = mock_embeddings("What is machine learning?")

        # Test that error is handled gracefully
        with pytest.raises(Exception, match=r"(API Error|OpenAI|Authentication)"):
            conversation_manager.answer_question("What is machine learning?")


def test_rag_pipeline_missing_document(rag_pipeline_factory, tmp_path):
    """Test RAG pipeline behavior with missing document file."""
    pipeline = rag_pipeline_factory("test_missing.db")
    missing_doc_path = tmp_path / "nonexistent.txt"

    # Test that missing document raises appropriate error
    with pytest.raises(
        (FileNotFoundError, Exception), match=r"(?i)(not found|no such file)"
    ):
        pipeline.process_document(missing_doc_path)


def test_conversation_history_persistence(
    rag_pipeline_factory,
    conversation_manager_factory,
    sample_embedded_chunks,
    conversation_manager_chat_mock_factory,
):
    """Test conversation history is maintained across multiple queries."""
    pipeline = rag_pipeline_factory("test_history.db")
    pipeline.vector_store.add_chunks(sample_embedded_chunks)
    conversation_manager = conversation_manager_factory(pipeline)

    # Mock successful RAG pipeline queries
    with patch.object(
        pipeline, "query", return_value=[(sample_embedded_chunks[0], 0.9)]
    ):
        # First query
        with conversation_manager_chat_mock_factory(
            conversation_manager, content="First response about ML"
        ):
            result1 = conversation_manager.answer_question("What is machine learning?")
            assert "answer" in result1
            assert len(conversation_manager.conversation_history) == 1

        # Second query should have access to previous context
        with conversation_manager_chat_mock_factory(
            conversation_manager, content="Follow-up response"
        ) as mock_chat:
            result2 = conversation_manager.answer_question("Can you elaborate?")
            assert "answer" in result2
            assert len(conversation_manager.conversation_history) == 2

            # Verify context was passed to the API call
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 1, "Should have one user message with full context"
            user_message = messages[0]
            assert "Previous Conversation" in user_message["content"]
            assert "What is machine learning?" in user_message["content"]


def test_multi_document_processing(rag_pipeline_factory, tmp_path, mock_embeddings):
    """Test processing multiple documents and querying across them."""
    pipeline = rag_pipeline_factory("test_multi_docs.db", chunk_size=100, overlap=20)

    test_documents = {
        "doc1.txt": "Machine learning is a powerful AI technique.",
        "doc2.txt": "Deep learning uses neural networks with many layers.",
        "doc3.txt": "Natural language processing helps machines understand text.",
    }
    doc_paths = create_test_documents(tmp_path, test_documents)

    def create_mock_chunk(text: str, source: str) -> object:
        return type(
            "DocumentChunk",
            (),
            {
                "content": text,
                "metadata": {"source": source, "chunk_id": 0},
                "embedding": None,
            },
        )()

    def mock_chunk_text_side_effect(text: str, source: str) -> list:
        return [create_mock_chunk(text, source)]

    with (
        patch.object(
            pipeline.chunker,
            "chunk_text",
            side_effect=mock_chunk_text_side_effect,
        ),
        patch.object(
            pipeline.embedding_service,
            "get_embeddings_batch",
            side_effect=lambda texts: [mock_embeddings(text) for text in texts],
        ),
        patch.object(
            pipeline.embedding_service,
            "get_embedding",
            return_value=mock_embeddings("AI techniques"),
        ),
    ):
        # Process all documents
        for doc_path in doc_paths.values():
            pipeline.process_document(doc_path)

        # Query and validate results
        results = pipeline.query("AI techniques")
        assert_valid_rag_result(results, min_chunks=0)

        # Verify vector store contains data from multiple documents
        assert pipeline.vector_store.embeddings is not None
        assert pipeline.vector_store.embeddings.shape[0] >= len(test_documents)


def test_large_document_chunking_integration(
    rag_pipeline_factory, large_document_setup, mock_embeddings
):
    """Test processing large documents with proper chunking and overlap."""
    _, doc_path, large_text = large_document_setup
    pipeline = rag_pipeline_factory("test_large.db", chunk_size=500, overlap=100)

    # Calculate expected number of chunks
    expected_min_chunks = len(large_text) // 400  # Conservative estimate

    with (
        patch.object(pipeline.embedding_service, "get_embeddings_batch") as mock_batch,
        patch.object(pipeline.embedding_service, "get_embedding") as mock_single,
    ):
        # Mock batch embedding to return embeddings for all chunks
        def mock_batch_func(texts: list) -> list:
            return [mock_embeddings(text) for text in texts]

        mock_batch.side_effect = mock_batch_func
        mock_single.return_value = mock_embeddings("machine learning test")

        # Process the large document
        pipeline.process_document(doc_path)

        # Verify proper chunking occurred
        assert pipeline.vector_store.embeddings is not None
        chunks_count = pipeline.vector_store.embeddings.shape[0]
        assert chunks_count >= expected_min_chunks

        # Test querying the large document
        results = pipeline.query("machine learning")
        assert len(results) > 0, "Should find relevant chunks in large document"
        assert all(-1.0 <= score <= 1.0 for _, score in results)


def test_conversation_manager_context_building(
    rag_pipeline_factory,
    conversation_manager_factory,
    sample_embedded_chunks,
    conversation_manager_chat_mock_factory,
):
    """Test that conversation manager properly builds context from retrieved chunks."""
    pipeline = rag_pipeline_factory("test_context.db")
    pipeline.vector_store.add_chunks(sample_embedded_chunks)
    conversation_manager = conversation_manager_factory(pipeline)

    # Mock pipeline to return specific chunks
    relevant_chunks = [
        (sample_embedded_chunks[0], 0.9),
        (sample_embedded_chunks[1], 0.7),
    ]

    with (
        patch.object(pipeline, "query", return_value=relevant_chunks),
        conversation_manager_chat_mock_factory(
            conversation_manager, content="Test response based on context"
        ) as mock_chat,
    ):
        result = conversation_manager.answer_question("What is machine learning?")

        # Verify the result structure
        assert_valid_conversation_result(result)
        assert len(result["retrieved_contexts"]) == 2

        # Verify API was called with proper context
        call_args = mock_chat.call_args
        messages = call_args.kwargs["messages"]

        # Check that context was built from retrieved chunks
        user_message = next(msg for msg in messages if msg["role"] == "user")
        context_content = user_message["content"]
        assert "Machine learning is a subset" in context_content
