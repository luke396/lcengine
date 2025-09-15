"""Data models for the RAG application."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""

    content: str
    metadata: dict[str, Any]
    embedding: np.ndarray | None = None


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""

    user_question: str
    bot_response: str
    retrieved_contexts: list[tuple[DocumentChunk, float]]
    timestamp: str
