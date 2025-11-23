"""OpenAI embeddings service."""

import numpy as np
from openai import OpenAI

from .config import config

logger = config.get_logger(__name__)


class EmbeddingService:
    """Handles OpenAI embeddings generation."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the EmbeddingService with OpenAI API key and model.

        Args:
            api_key: OpenAI API key. If None,
                reads from OPENAI_API_KEY environment variable.
            model: Embedding model name. If None, uses config.EMBEDDING_MODEL.
        """
        api_key = api_key or config.get_openai_api_key()
        default_headers = config.get_api_headers()
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.OPENAI_BASE_URL,
            default_headers=default_headers or None,
        )
        self.model = model or config.EMBEDDING_MODEL

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            np.ndarray: The embedding vector for the input text.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = np.array(response.data[0].embedding)
        except Exception:
            logger.exception("Error generating embedding")
            raise
        else:
            return embedding

    def get_embeddings_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[np.ndarray]:
        """Get embeddings for multiple texts in batches.

        Args:
            texts: List of input texts to generate embeddings for.
            batch_size: Number of texts to process in each batch.

        Returns:
            list[np.ndarray]: List of embedding vectors for the input texts.
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                embeddings.extend(batch_embeddings)
                logger.info("Generated embeddings for batch %d", i // batch_size + 1)
            except Exception:
                logger.exception("Error generating batch embeddings")
                raise

        return embeddings
