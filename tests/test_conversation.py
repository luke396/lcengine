"""Comprehensive tests for ConversationManager class."""

import datetime
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from app import ConversationManager, ConversationTurn, DocumentChunk


@pytest.mark.parametrize(
    ("init_kwargs", "expected_key", "requires_env_patch"),
    [
        ({"openai_api_key": "test-key"}, "test-key", False),
        ({}, "env-key", True),
    ],
)
def test_conversation_manager_initialization(
    mock_rag_pipeline_empty_return, init_kwargs, expected_key, requires_env_patch
):
    with ExitStack() as stack:
        if requires_env_patch:
            stack.enter_context(
                patch(
                    "app.conversation.config.get_openai_api_key",
                    return_value=expected_key,
                )
            )
        manager = ConversationManager(
            rag_pipeline=mock_rag_pipeline_empty_return, **init_kwargs
        )

    assert manager.rag_pipeline == mock_rag_pipeline_empty_return
    assert manager.client.api_key == expected_key
    assert manager.conversation_history == []
    assert manager.max_history_turns == 5


def test_conversation_history_storage(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    with (
        patch.object(
            conversation_manager, "generate_standalone_query", return_value="test query"
        ),
        patch.object(
            conversation_manager.rag_pipeline, "query", return_value=sample_chunks
        ),
        conversation_manager_chat_mock_factory(conversation_manager, "Test answer"),
    ):
        _ = conversation_manager.answer_question("What is AI?")

        assert len(conversation_manager.conversation_history) == 1
        turn = conversation_manager.conversation_history[0]
        assert turn.user_question == "What is AI?"
        assert turn.bot_response == "Test answer"
        assert len(turn.retrieved_contexts) == len(sample_chunks)


def test_conversation_history_limit(conversation_manager, sample_chunks):
    for i in range(7):
        turn = ConversationTurn(
            user_question=f"Question {i}",
            bot_response=f"Answer {i}",
            retrieved_contexts=sample_chunks,
            timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )
        conversation_manager.conversation_history.append(turn)

    # Test that context building only uses the last 5 turns
    context_prompt = conversation_manager.build_context_prompt(
        "Test question", sample_chunks
    )

    # Should only contain the last 5 questions (2, 3, 4, 5, 6)
    assert "Question 0" not in context_prompt
    assert "Question 1" not in context_prompt
    assert "Question 2" in context_prompt
    assert "Question 6" in context_prompt


def test_clear_history(conversation_manager, sample_chunks):
    turn = ConversationTurn(
        user_question="Test question",
        bot_response="Test answer",
        retrieved_contexts=sample_chunks,
        timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )
    conversation_manager.conversation_history.append(turn)

    assert len(conversation_manager.conversation_history) == 1

    conversation_manager.clear_history()

    assert len(conversation_manager.conversation_history) == 0


def test_standalone_query_first_turn(conversation_manager):
    question = "What is machine learning?"
    result = conversation_manager.generate_standalone_query(question)
    assert result == question


def test_standalone_query_with_context(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    turn = ConversationTurn(
        user_question="What is AI?",
        bot_response="AI is artificial intelligence.",
        retrieved_contexts=sample_chunks,
        timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )
    conversation_manager.conversation_history.append(turn)

    with conversation_manager_chat_mock_factory(
        conversation_manager, "What are the applications of artificial intelligence?"
    ) as mock_create:
        result = conversation_manager.generate_standalone_query(
            "What are its applications?"
        )

        assert result == "What are the applications of artificial intelligence?"
        mock_create.assert_called_once()

        call_args = mock_create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Given the following conversation history" in prompt_content
        assert "What is AI?" in prompt_content
        assert "What are its applications?" in prompt_content


def test_standalone_query_multiple_turns(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    for i in range(5):
        turn = ConversationTurn(
            user_question=f"Question {i}",
            bot_response=f"Answer {i}",
            retrieved_contexts=sample_chunks,
            timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )
        conversation_manager.conversation_history.append(turn)

    with conversation_manager_chat_mock_factory(
        conversation_manager, "Rewritten question"
    ) as mock_create:
        conversation_manager.generate_standalone_query("Follow-up question")

        # Verify only last 3 turns are in the prompt
        call_args = mock_create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "Question 0" not in prompt_content
        assert "Question 1" not in prompt_content
        assert "Question 2" in prompt_content
        assert "Question 3" in prompt_content
        assert "Question 4" in prompt_content


def test_standalone_query_api_error_handling(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    turn = ConversationTurn(
        user_question="What is AI?",
        bot_response="AI is artificial intelligence.",
        retrieved_contexts=sample_chunks,
        timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )
    conversation_manager.conversation_history.append(turn)

    with (
        conversation_manager_chat_mock_factory(
            conversation_manager, side_effect=ValueError("API Error")
        ),
        pytest.raises(ValueError, match="API Error"),
    ):
        conversation_manager.generate_standalone_query("Follow-up question")


def test_standalone_query_empty_response(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    turn = ConversationTurn(
        user_question="What is AI?",
        bot_response="AI is artificial intelligence.",
        retrieved_contexts=sample_chunks,
        timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )
    conversation_manager.conversation_history.append(turn)

    with conversation_manager_chat_mock_factory(conversation_manager, None):
        result = conversation_manager.generate_standalone_query("Follow-up question")

        # Should fall back to original question
        assert result == "Follow-up question"


def test_build_context_prompt_without_history(conversation_manager, sample_chunks):
    question = "What is machine learning?"
    prompt = conversation_manager.build_context_prompt(question, sample_chunks)

    assert prompt.startswith("You are a helpful assistant")
    assert "Previous Conversation" not in prompt
    assert "Relevant Document Sections" in prompt
    assert question in prompt
    assert "Current Question: " + question in prompt
    assert "1. Base your answer primarily on the provided document sections" in prompt
    assert prompt.endswith(
        "Please provide a helpful and accurate response based on the context above:"
    )

    for chunk, score in sample_chunks:
        assert chunk.content in prompt
        assert f"{score:.4f}" in prompt


def test_build_context_prompt_with_history(conversation_manager, sample_chunks):
    turn = ConversationTurn(
        user_question="What is AI?",
        bot_response="AI is artificial intelligence.",
        retrieved_contexts=sample_chunks,
        timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )
    conversation_manager.conversation_history.append(turn)

    question = "How does it work?"
    prompt = conversation_manager.build_context_prompt(question, sample_chunks)

    assert "=== Previous Conversation ===" in prompt
    assert "Human: What is AI?" in prompt
    assert "Assistant: AI is artificial intelligence." in prompt
    assert "=== Relevant Document Sections ===" in prompt
    assert question in prompt


def test_context_prompt_chunk_formatting(conversation_manager):
    chunks = [
        (
            DocumentChunk(
                content="Machine learning content",
                metadata={"source": "ml_doc.pdf", "chunk_id": 1},
            ),
            0.8542,
        ),
        (
            DocumentChunk(
                content="Deep learning content",
                metadata={"source": "dl_doc.pdf", "chunk_id": 2},
            ),
            0.7234,
        ),
    ]

    prompt = conversation_manager.build_context_prompt("Test question", chunks)

    assert "[Context 1] (Similarity: 0.8542)" in prompt
    assert "Source: ml_doc.pdf" in prompt
    assert "Content: Machine learning content" in prompt
    assert "[Context 2] (Similarity: 0.7234)" in prompt
    assert "Source: dl_doc.pdf" in prompt
    assert "Content: Deep learning content" in prompt


def test_answer_question_success_flow(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    with (
        patch.object(
            conversation_manager,
            "generate_standalone_query",
            return_value="standalone question",
        ),
        patch.object(
            conversation_manager.rag_pipeline, "query", return_value=sample_chunks
        ),
        conversation_manager_chat_mock_factory(
            conversation_manager, "Test answer content"
        ),
    ):
        result = conversation_manager.answer_question("What is AI?")

        assert result["answer"] == "Test answer content"
        assert result["retrieved_contexts"] == sample_chunks
        assert "confidence" in result
        assert "standalone_query" in result
        assert result["standalone_query"] == "standalone question"


def test_answer_question_no_retrieved_chunks(conversation_manager):
    with (
        patch.object(
            conversation_manager, "generate_standalone_query", return_value="test query"
        ),
        patch.object(conversation_manager.rag_pipeline, "query", return_value=[]),
    ):
        result = conversation_manager.answer_question("What is AI?")

        assert "I don't have enough information" in result["answer"]
        assert result["retrieved_contexts"] == []
        assert result["confidence"] == 0.0


def test_answer_question_with_conversation_context(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    turn = ConversationTurn(
        user_question="What is machine learning?",
        bot_response="ML is a subset of AI.",
        retrieved_contexts=sample_chunks,
        timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
    )
    conversation_manager.conversation_history.append(turn)

    with (
        patch.object(
            conversation_manager.rag_pipeline, "query", return_value=sample_chunks
        ),
        conversation_manager_chat_mock_factory(
            conversation_manager, "Context-aware answer"
        ) as mock_create,
    ):
        _ = conversation_manager.answer_question("How does it work?")

        call_args = mock_create.call_args
        prompt_content = call_args[1]["messages"][0]["content"]
        assert "What is machine learning?" in prompt_content
        assert "ML is a subset of AI." in prompt_content


def test_confidence_score_calculation(
    conversation_manager, conversation_manager_chat_mock_factory
):
    chunks_with_scores = [
        (DocumentChunk(content="content1", metadata={"source": "doc1"}), 0.9),
        (DocumentChunk(content="content2", metadata={"source": "doc2"}), 0.7),
        (DocumentChunk(content="content3", metadata={"source": "doc3"}), 0.5),
    ]

    with (
        patch.object(
            conversation_manager, "generate_standalone_query", return_value="test"
        ),
        patch.object(
            conversation_manager.rag_pipeline, "query", return_value=chunks_with_scores
        ),
        conversation_manager_chat_mock_factory(conversation_manager, "Answer"),
    ):
        result = conversation_manager.answer_question("Test question")

        expected_confidence = np.mean([0.9, 0.7, 0.5])
        assert abs(result["confidence"] - expected_confidence) < 1e-6


def test_answer_question_api_error_handling(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    with (
        patch.object(
            conversation_manager, "generate_standalone_query", return_value="test"
        ),
        patch.object(
            conversation_manager.rag_pipeline, "query", return_value=sample_chunks
        ),
        conversation_manager_chat_mock_factory(
            conversation_manager, side_effect=ValueError("API parsing error")
        ),
    ):
        result = conversation_manager.answer_question("Test question")

        assert "I encountered a parsing error" in result["answer"]
        assert result["retrieved_contexts"] == sample_chunks
        assert result["confidence"] == 0.0


def test_answer_question_empty_response(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    with (
        patch.object(
            conversation_manager, "generate_standalone_query", return_value="test"
        ),
        patch.object(
            conversation_manager.rag_pipeline, "query", return_value=sample_chunks
        ),
        conversation_manager_chat_mock_factory(conversation_manager, None),
    ):
        result = conversation_manager.answer_question("Test question")

        assert result["answer"] == "I apologize, but I couldn't generate a response."


def test_invalid_query_handling(conversation_manager):
    result = conversation_manager.answer_question("")

    assert isinstance(result, dict)
    assert "answer" in result


def test_rag_pipeline_error_handling(conversation_manager):
    with (
        patch.object(
            conversation_manager, "generate_standalone_query", return_value="test"
        ),
        patch.object(conversation_manager.rag_pipeline, "query") as mock_query,
    ):
        mock_query.side_effect = RuntimeError("RAG pipeline error")

        with pytest.raises(RuntimeError):
            conversation_manager.answer_question("Test question")


def test_conversation_turn_creation(
    conversation_manager, sample_chunks, conversation_manager_chat_mock_factory
):
    with (
        patch.object(
            conversation_manager, "generate_standalone_query", return_value="test"
        ),
        patch.object(
            conversation_manager.rag_pipeline, "query", return_value=sample_chunks
        ),
        conversation_manager_chat_mock_factory(conversation_manager, "Test answer"),
        patch("app.conversation.datetime") as mock_datetime,
    ):
        mock_datetime.datetime.now.return_value.isoformat.return_value = (
            "2023-01-01T00:00:00Z"
        )
        mock_datetime.UTC = datetime.UTC

        conversation_manager.answer_question("Test question")

        turn = conversation_manager.conversation_history[0]
        assert turn.timestamp == "2023-01-01T00:00:00Z"
