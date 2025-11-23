"""Conversation management with context building and history."""

import datetime
from typing import Any

import numpy as np
from openai import OpenAI

from .config import config
from .models import ConversationTurn, DocumentChunk
from .pipeline import RAGPipeline

logger = config.get_logger(__name__)


class ConversationManager:
    """Manages multi-round conversations with context building."""

    def __init__(
        self, rag_pipeline: RAGPipeline, openai_api_key: str | None = None
    ) -> None:
        """Initialize ConversationManager.

        Args:
            rag_pipeline: RAG pipeline instance.
            openai_api_key: OpenAI API key.
        """
        self.rag_pipeline: RAGPipeline = rag_pipeline
        default_headers = config.get_api_headers()
        self.client = OpenAI(
            api_key=openai_api_key or config.get_openai_api_key(),
            base_url=config.OPENAI_BASE_URL,
            default_headers=default_headers or None,
        )
        self.conversation_history: list[ConversationTurn] = []
        self.max_history_turns = 5

    def build_context_prompt(
        self,
        question: str,
        retrieved_chunks: list[tuple[DocumentChunk, float]],
    ) -> str:
        """Build context-aware prompt with conversation history and retrieved documents.

        Returns:
            str: The constructed prompt string containing conversation history
                and relevant document sections.
        """
        history_context = ""
        if self.conversation_history:
            history_context = "\n\n=== Previous Conversation ===\n"
            for turn in self.conversation_history[-self.max_history_turns :]:
                history_context += (
                    f"Human: {turn.user_question}\nAssistant: {turn.bot_response}\n\n"
                )

        doc_context = "\n=== Relevant Document Sections ===\n"
        for i, (chunk, score) in enumerate(retrieved_chunks):
            doc_context += (
                f"\n[Context {i + 1}] (Similarity: {score:.4f})\n"
                f"Source: {chunk.metadata['source']}\n"
                f"Content: {chunk.content}\n"
            )

        return (
            f"You are a helpful assistant that answers questions based on the provided "
            f"document context and conversation history.\n\n"
            "Use the following guidelines:\n"
            "1. Base your answer primarily on the provided document sections\n"
            "2. Consider the conversation history to maintain context continuity\n"
            "3. If the answer is not in the documents, acknowledge this limitation\n"
            "4. Be conversational and reference previous questions when relevant\n"
            "5. Provide specific details from the documents when possible\n\n"
            f"{history_context}\n\n"
            f"{doc_context}\n\n"
            f"Current Question: {question}\n\n"
            "Please provide a helpful and accurate response based on the context above:"
        )

    def generate_standalone_query(self, question: str) -> str:
        """Generate a standalone query from conversation context for better retrieval.

        Returns:
            str: The standalone question generated from the conversation context.
        """
        if not self.conversation_history:
            return question

        context = ""
        for turn in self.conversation_history[-3:]:  # Last 3 turns
            context += f"Human: {turn.user_question}\nAssistant: {turn.bot_response}\n"

        prompt = (
            "Given the following conversation history and a follow-up question, "
            "rewrite the follow-up question as a standalone question that can be "
            "understood without the conversation context.\n\n"
            f"Conversation History:\n{context}\n\n"
            f"Follow-up Question: {question}\n\n"
            "Standalone Question:"
        )

        response = self.client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.QUERY_REWRITE_MAX_TOKENS,
            temperature=config.QUERY_REWRITE_TEMPERATURE,
        )
        standalone_query = response.choices[0].message.content
        standalone_query = standalone_query.strip() if standalone_query else question
        logger.info("Generated standalone query: %s", standalone_query)
        return standalone_query

    def answer_question(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Answer a question using RAG with conversation context.

        Returns:
            dict[str, Any]: A dictionary containing the answer, retrieved contexts,
                confidence score, and the standalone query used for retrieval.
        """
        logger.info("Processing question: %s", question)

        standalone_query = self.generate_standalone_query(question)
        retrieved_chunks = self.rag_pipeline.query(standalone_query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": (
                    "I don't have enough information "
                    "in the uploaded document to answer "
                    "your question. Please try uploading a relevant document first."
                ),
                "retrieved_contexts": [],
                "confidence": 0.0,
            }

        context_prompt = self.build_context_prompt(question, retrieved_chunks)

        try:
            response = self.client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": context_prompt}],
                max_tokens=config.CHAT_MAX_TOKENS,
                temperature=config.CHAT_TEMPERATURE,
            )
            answer = response.choices[0].message.content
            if answer:
                answer = answer.strip()
            else:
                answer = "I apologize, but I couldn't generate a response."

            avg_score = np.mean([score for _, score in retrieved_chunks])
            confidence = float(avg_score)

            turn = ConversationTurn(
                user_question=question,
                bot_response=answer,
                retrieved_contexts=retrieved_chunks,
                timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
            )
            self.conversation_history.append(turn)

            logger.info("Retrieved contexts:")
            for i, (chunk, score) in enumerate(retrieved_chunks):
                logger.info(
                    "  Context %d: %s (score: %.4f)",
                    i + 1,
                    chunk.metadata["source"],
                    score,
                )
                logger.info("  Preview: %s...", chunk.content[:100])

        except ValueError as e:
            logger.exception("Error parsing response: %s")
            return {
                "answer": (
                    "I encountered a parsing error while processing your question: "
                    f"{e!s}"
                ),
                "retrieved_contexts": retrieved_chunks,
                "confidence": 0.0,
            }
        else:
            return {
                "answer": answer,
                "retrieved_contexts": retrieved_chunks,
                "confidence": confidence,
                "standalone_query": standalone_query,
            }

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")
