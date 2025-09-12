"""Unit tests for document processing components."""

from pathlib import Path

import pytest

from app.core import DocumentLoader, TextChunker


def test_load_txt_document(sample_document_path):
    """Test loading TXT document."""
    if not sample_document_path.exists():
        pytest.skip("Sample document not found")

    text = DocumentLoader.load_document(sample_document_path)

    assert isinstance(text, str)
    assert len(text) > 0
    assert "Machine Learning" in text


def test_load_nonexistent_file():
    """Test loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load_document(Path("nonexistent_file.txt"))


def test_unsupported_file_type():
    """Test unsupported file type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentLoader.load_document(Path("test.invalid"))


def test_chunk_creation():
    """Test basic text chunking."""
    chunker = TextChunker(chunk_size=100, overlap=20)
    text = "This is a test document. " * 20  # Make text long enough

    chunks = chunker.chunk_text(text, source="test_doc")

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content) > 0
        assert chunk.metadata["source"] == "test_doc"
        assert "chunk_id" in chunk.metadata
        assert "start_char" in chunk.metadata
        assert "end_char" in chunk.metadata


def test_empty_text_chunking():
    """Test chunking empty text."""
    chunker = TextChunker()
    chunks = chunker.chunk_text("", "empty_source")

    assert len(chunks) == 0


def test_chunk_overlap():
    """Test that chunks have proper overlap."""
    chunker = TextChunker(chunk_size=50, overlap=10)
    text = "A" * 100  # Simple text for testing

    chunks = chunker.chunk_text(text, "test")

    # Check overlap exists
    first_end = chunks[0].metadata["end_char"]
    second_start = chunks[1].metadata["start_char"]
    assert first_end - second_start == 10  # Expected overlap


def test_chunk_metadata_completeness():
    """Test that chunk metadata is complete."""
    chunker = TextChunker(chunk_size=100, overlap=20)
    text = "Test text for metadata checking. " * 10

    chunks = chunker.chunk_text(text, "metadata_test")

    required_fields = ["source", "chunk_id", "start_char", "end_char", "length"]
    for chunk in chunks:
        for field in required_fields:
            assert field in chunk.metadata
            assert chunk.metadata[field] is not None
