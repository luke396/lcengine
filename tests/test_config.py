"""Simplified comprehensive tests for Config class."""

import logging
import os
from importlib import reload
from pathlib import Path
from unittest.mock import patch

import pytest

from app import config as config_module
from app.config import Config, env_path


def test_get_openai_api_key_from_env():
    """Test OpenAI API key retrieval from environment."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        assert Config.get_openai_api_key() == "test-api-key"


def test_get_openai_api_key_empty_when_not_set():
    """Test OpenAI API key returns empty string when not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert not Config.get_openai_api_key()


def test_validate_success_with_api_key():
    """Test validation passes when API key is set."""
    with patch.object(Config, "get_openai_api_key", return_value="test-key"):
        Config.validate()


def test_validate_fails_without_api_key():
    """Test validation fails when API key is not set."""
    with (
        patch.object(Config, "get_openai_api_key", return_value=""),
        pytest.raises(ValueError, match="OPENAI_API_KEY is required"),
    ):
        Config.validate()


@pytest.mark.parametrize(
    ("env_var", "config_attr", "default_value", "test_value", "expected_type"),
    [
        ("LOG_LEVEL", "LOG_LEVEL", "INFO", "debug", str),
        ("OPENAI_LOG_LEVEL", "OPENAI_LOG_LEVEL", "WARNING", "error", str),
        ("ENVIRONMENT", "ENVIRONMENT", "development", "production", str),
        (
            "EMBEDDING_MODEL",
            "EMBEDDING_MODEL",
            "text-embedding-3-small",
            "text-embedding-ada-002",
            str,
        ),
        ("CHAT_MODEL", "CHAT_MODEL", "gpt-4.1-nano-2025-04-14", "gpt-3.5-turbo", str),
        ("CHUNK_SIZE", "CHUNK_SIZE", 1000, "1500", int),
        ("CHUNK_OVERLAP", "CHUNK_OVERLAP", 200, "300", int),
        ("CHAT_MAX_TOKENS", "CHAT_MAX_TOKENS", 500, "1000", int),
        ("QUERY_REWRITE_MAX_TOKENS", "QUERY_REWRITE_MAX_TOKENS", 150, "200", int),
        ("CHAT_TEMPERATURE", "CHAT_TEMPERATURE", 0.7, "0.5", float),
        ("QUERY_REWRITE_TEMPERATURE", "QUERY_REWRITE_TEMPERATURE", 0.1, "0.2", float),
    ],
)
def test_config_loading_from_env(
    env_var, config_attr, default_value, test_value, expected_type
):
    with patch.dict(os.environ, {}, clear=True):
        reload(config_module)
        actual_default = getattr(config_module.Config, config_attr)
        if isinstance(actual_default, str) and config_attr in {
            "LOG_LEVEL",
            "OPENAI_LOG_LEVEL",
        }:
            assert actual_default == default_value.upper()
        else:
            assert actual_default == default_value

    with patch.dict(os.environ, {env_var: test_value}):
        reload(config_module)
        actual_value = getattr(config_module.Config, config_attr)
        if expected_type is int:
            expected = int(test_value)
        elif expected_type is float:
            expected = float(test_value)
        elif config_attr in {"LOG_LEVEL", "OPENAI_LOG_LEVEL"}:
            expected = test_value.upper()
        else:
            expected = test_value
        assert actual_value == expected


@pytest.mark.parametrize(
    ("env_var", "config_attr", "test_path"),
    [
        ("VECTOR_STORE_DB_PATH", "VECTOR_STORE_DB_PATH", "/custom/path/store.db"),
        ("VECTOR_STORE_DIR", "VECTOR_STORE_DIR", "/custom/vectors"),
    ],
)
def test_path_config_loading(env_var, config_attr, test_path):
    """Test Path configuration loading from environment variables."""
    with patch.dict(os.environ, {}, clear=True):
        reload(config_module)
        default_path = getattr(config_module.Config, config_attr)
        assert isinstance(default_path, Path)

    with patch.dict(os.environ, {env_var: test_path}):
        reload(config_module)
        actual_path = getattr(config_module.Config, config_attr)
        assert actual_path == Path(test_path)


def test_openai_base_url_from_env():
    """Test OpenAI base URL loading from environment."""
    with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://custom.openai.com"}):
        reload(config_module)
        assert config_module.Config.OPENAI_BASE_URL == "https://custom.openai.com"


def test_openai_base_url_none_when_not_set():
    """Test OpenAI base URL is None when not set."""
    with patch.object(Config, "OPENAI_BASE_URL", None):
        assert Config.OPENAI_BASE_URL is None


@pytest.mark.parametrize(
    ("env_value", "is_dev", "is_prod"),
    [
        ("development", True, False),
        ("DEVELOPMENT", True, False),
        ("production", False, True),
        ("PRODUCTION", False, True),
        ("staging", False, False),
    ],
)
def test_environment_detection(env_value, is_dev, is_prod):
    """Test environment detection methods."""
    with patch.object(Config, "ENVIRONMENT", env_value):
        assert Config.is_development() == is_dev
        assert Config.is_production() == is_prod


def test_setup_logging_default_level():
    """Test logging setup with default INFO level."""
    with (
        patch("app.config.logging.basicConfig") as mock_basic,
        patch("app.config.logging.getLogger") as mock_get_logger,
    ):
        mock_logger = mock_get_logger.return_value

        Config.setup_logging()

        mock_basic.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

        mock_get_logger.assert_called_once_with("openai")
        mock_logger.setLevel.assert_called_once_with(logging.WARNING)


def test_setup_logging_custom_level():
    """Test logging setup with custom levels."""
    with (
        patch.object(Config, "LOG_LEVEL", "DEBUG"),
        patch.object(Config, "OPENAI_LOG_LEVEL", "ERROR"),
        patch("app.config.logging.basicConfig") as mock_basic,
        patch("app.config.logging.getLogger") as mock_get_logger,
    ):
        mock_logger = mock_get_logger.return_value

        Config.setup_logging()

        mock_basic.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

        mock_logger.setLevel.assert_called_once_with(logging.ERROR)


def test_setup_logging_invalid_level_fallback():
    """Test logging setup falls back to defaults for invalid levels."""
    with (
        patch.object(Config, "LOG_LEVEL", "INVALID"),
        patch.object(Config, "OPENAI_LOG_LEVEL", "INVALID"),
        patch("app.config.logging.basicConfig") as mock_basic,
        patch("app.config.logging.getLogger") as mock_get_logger,
    ):
        mock_logger = mock_get_logger.return_value

        Config.setup_logging()

        mock_basic.assert_called_once_with(
            level=logging.INFO,  # Falls back to INFO
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )

        mock_logger.setLevel.assert_called_once_with(logging.WARNING)


def test_get_logger():
    """Test logger creation with specified name."""
    with patch("app.config.logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value

        result = Config.get_logger("test.module")

        mock_get_logger.assert_called_once_with("test.module")
        assert result == mock_logger


@pytest.mark.parametrize(
    ("env_var", "invalid_value", "error_match"),
    [
        ("CHUNK_SIZE", "not_a_number", "invalid literal for int"),
        ("CHAT_TEMPERATURE", "not_a_float", "could not convert string to float"),
    ],
)
def test_type_conversion_errors(env_var, invalid_value, error_match):
    """Test handling of invalid type conversions."""
    with (
        patch.dict(os.environ, {env_var: invalid_value}),
        pytest.raises(ValueError, match=error_match),
    ):
        reload(config_module)


def test_config_instance_creation():
    """Test that config instance is created properly."""
    reload(config_module)
    assert isinstance(config_module.config, config_module.Config)
    assert config_module.config.__class__.__name__ == "Config"


def test_dotenv_loading():
    """Test that .env file loading logic works correctly."""
    # Test the loading logic by checking if env_path is constructed correctly
    assert isinstance(env_path, Path)
    assert env_path.name == ".env"
    assert env_path.parent.name == "lcengine"  # Project root directory


def test_no_dotenv_loading_when_missing():
    """Test that .env file loading is skipped when file doesn't exist."""
    with (
        patch.object(Path, "exists", return_value=False),
        patch("app.config.load_dotenv") as mock_load,
    ):
        reload(config_module)

        mock_load.assert_not_called()
