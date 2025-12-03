"""Configuration management for LCEngine application."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"

if env_path.exists():
    load_dotenv(env_path)


class Config:
    """Application configuration loaded from environment variables."""

    # OpenAI Configuration
    @classmethod
    def get_openai_api_key(cls) -> str:
        """Get OpenAI API key from environment variables.

        Returns:
            OpenAI API key from environment or empty string if not set.
        """
        return os.getenv("OPENAI_API_KEY", "")

    OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    OPENAI_LOG_LEVEL: str = os.getenv("OPENAI_LOG_LEVEL", "WARNING").upper()

    # Application Settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Chat Model Configuration
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4.1-nano-2025-04-14")
    CHAT_MAX_TOKENS: int = int(os.getenv("CHAT_MAX_TOKENS", "500"))
    CHAT_TEMPERATURE: float = float(os.getenv("CHAT_TEMPERATURE", "0.7"))

    # Query Rewriting Configuration
    QUERY_REWRITE_MAX_TOKENS: int = int(os.getenv("QUERY_REWRITE_MAX_TOKENS", "150"))
    QUERY_REWRITE_TEMPERATURE: float = float(
        os.getenv("QUERY_REWRITE_TEMPERATURE", "0.1")
    )

    # Vector Store Configuration
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "faiss").lower()
    VECTOR_STORE_DB_PATH: Path = Path(
        os.getenv("VECTOR_STORE_DB_PATH", "data/vector_store.db")
    )
    VECTOR_STORE_DIR: Path = Path(os.getenv("VECTOR_STORE_DIR", "data/vectors"))
    FAISS_INDEX_PATH: Path = Path(
        os.getenv("FAISS_INDEX_PATH", "data/faiss/index.faiss")
    )
    VECTOR_RAW_TOP_K_MULTIPLIER: int = int(
        os.getenv("VECTOR_RAW_TOP_K_MULTIPLIER", "2")
    )

    # API Header Configuration
    API_USER_AGENT: str = os.getenv("API_USER_AGENT", "LCEngine/1.0")
    API_TEST_HEADER_NAME: str | None = os.getenv(
        "API_TEST_HEADER_NAME",
        "X-LCEngine-Test-Token",
    )
    API_TEST_HEADER_VALUE: str | None = os.getenv(
        "API_TEST_HEADER_VALUE",
        "allow",
    )

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration values.

        Raises:
            ValueError: If OPENAI_API_KEY is not set.
        """
        if not cls.get_openai_api_key():
            msg = (
                "OPENAI_API_KEY is required. Please set it in .env file or environment."
            )
            raise ValueError(msg)

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development environment.

        Returns:
            True if environment is development.
        """
        return cls.ENVIRONMENT.lower() == "development"

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment.

        Returns:
            True if environment is production.
        """
        return cls.ENVIRONMENT.lower() == "production"

    @classmethod
    def setup_logging(cls) -> None:
        """Setup basic logging configuration.

        Configure logging once at application startup with:
        - Console output for all levels
        - Simple, readable format
        - Configurable level via environment variable
        """
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

        # Configure third-party library log levels via environment variables
        logging.getLogger("openai").setLevel(
            getattr(logging, cls.OPENAI_LOG_LEVEL, logging.WARNING)
        )

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger with the specified name.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Logger instance
        """
        return logging.getLogger(name)

    @classmethod
    def get_api_headers(cls) -> dict[str, str]:
        """Build default headers for outbound API calls.

        Returns:
            Mapping of header names to values used on outbound HTTP requests.
        """
        headers: dict[str, str] = {}

        if cls.API_USER_AGENT:
            headers["User-Agent"] = cls.API_USER_AGENT

        if cls.API_TEST_HEADER_NAME and cls.API_TEST_HEADER_VALUE:
            headers[cls.API_TEST_HEADER_NAME] = cls.API_TEST_HEADER_VALUE

        return headers


config = Config()
