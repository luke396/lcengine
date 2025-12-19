# Development guidelines for LCEngine

This document provides context for AI coding assistants working on the LCEngine RAG system.

## Project architecture and context

### Repository structure

This is a Python RAG application using native implementation for core algorithms.

```txt
lcengine/
├── app/                      # Core application modules
│   ├── config.py            # Configuration management and environment validation
│   ├── models.py            # Data models (DocumentChunk, ConversationTurn)
│   ├── embeddings.py        # OpenAI embeddings service wrapper
│   ├── document_processing.py  # PDF/TXT loading and chunking strategies
│   ├── vector_store/        # Vector storage implementations
│   │   ├── sqlite_store.py  # SQLite + NumPy backend (v0.1 baseline)
│   │   └── faiss_store.py   # FAISS integration (v0.2+, production default)
│   ├── pipeline.py          # RAG pipeline orchestration (retrieval + generation)
│   └── conversation.py      # Multi-turn dialogue management
├── tests/                   # Comprehensive test suite (85%+ coverage)
│   ├── conftest.py          # Pytest fixtures and mocks
│   ├── test_*.py            # Unit and integration tests
│   └── data/                # Test datasets and evaluation fixtures
├── docs/                    # Design documentation and experiments
│   ├── v0.2_design.md       # Architecture decisions for v0.2 (FAISS, memory)
│   ├── ROADMAP.md           # Development roadmap (v0.2-v0.6)
│   └── experiments.md       # Evaluation results and metric comparisons (planned)
├── app.py                   # Streamlit UI entry point
├── main.py                  # CLI launcher with logging setup
├── evaluate.py              # Evaluation script (RAGAS + Hit@k metrics)
└── pyproject.toml           # Project metadata and dependencies
```

- **Core layer** (`app/`): Native RAG implementation with pluggable vector stores
- **Storage layer** (`app/vector_store/`): Abstract interface with FAISS (production) and SQLite (fallback) backends
- **UI layer** (`app.py`): Streamlit interface for document upload and chat
- **Testing layer** (`tests/`): Fast unit tests + performance benchmarks (`@pytest.mark.benchmark`)

### Development tools & commands

- `uv` – Fast Python package installer (replaces pip/poetry)
- `pytest` – Testing framework with coverage tracking
- `ruff` – Fast Python linter and formatter
- `pyright` – Static type checking
- `streamlit` – UI framework for rapid prototyping

This project uses `uv` for dependency management and `pyproject.toml` for configuration.

```bash
# Install/update dependencies
uv sync

# Run application
python main.py

# Run tests (skip performance benchmarks)
uv run pytest -m "not benchmark"

# Run with coverage
uv run pytest --cov=app --cov-report=term-missing

# Lint and format
ruff check app tests
ruff format app tests

# Type checking
pyright app tests

# Run evaluation
python evaluate.py --dataset tests/data/evaluation_dataset.json --k 5
```

#### Key config files

- `pyproject.toml`: Dependencies and tool configuration
- `uv.lock`: Locked dependencies for reproducible builds
- `.env`: API keys and runtime configuration (gitignored, see `.env.example`)

#### Commit standards

Commits follow Conventional Commit prefixes (`feat:`, `fix:`, `test:`, `chore:`) with concise scopes. Keep commit messages to a single line describing the change. Detailed explanations should be saved for the pull request description.

#### Pull request guidelines

For pull requests include a summary, linked issue or roadmap item, test command output, configuration updates (e.g., new env vars), and visuals or transcripts when the UI changes.

## Core development principles

### Test-Driven Development (TDD)

CRITICAL: Always write tests before implementation. No exceptions.

**TDD workflow (Red-Green-Refactor):**

1. **Red**: Write a failing test for the new functionality
2. **Green**: Write the minimal code to make the test pass
3. **Refactor**: Improve code quality while keeping tests green

Ask before writing any implementation code: "Have I written the test first?"

### Keep it simple and minimal

CRITICAL: Prioritize simplicity over extensibility. Implement only what is needed for current requirements.

**Guidelines:**

- No premature abstractions or "future-proofing"
- No backward compatibility layers (project is pre-1.0)
- No complex inheritance hierarchies
- Prefer simple functions over elaborate class structures
- Delete unused code immediately

Ask: "Is this the simplest solution that works?"

### Code quality standards

All Python code MUST include type hints and return types.

```python
def filter_chunks(chunks: list[DocumentChunk], min_score: float) -> list[DocumentChunk]:
    """Filter chunks by minimum similarity score.

    Args:
        chunks: List of document chunks with similarity scores.
        min_score: Minimum score threshold (0.0 to 1.0).

    Returns:
        Filtered list of chunks above the threshold.
    """
```

- Use four-space indentation, PEP 8 layout
- Follow Google-style docstrings (enforced by Ruff)
- Use descriptive variable names (`chunk_size` not `cs`)
- Keep functions small and focused (<30 lines when possible)

### Testing requirements

**Test categories:**

- Unit tests: `tests/test_*.py` (default, no marker needed - fast, isolated tests)
- Integration tests: Use `@pytest.mark.integration` (multi-module collaboration tests)
- Performance benchmarks: Use `@pytest.mark.benchmark` (large data volume, latency measurement)
- Maintain 85%+ coverage (verify with `pytest --cov=app`)

**Running tests:**

```bash
# Fast unit tests only (default)
pytest

# Include integration tests
pytest -m integration

# Run performance benchmarks
pytest -m benchmark

# Run all tests except benchmarks
pytest -m "not benchmark"
```

**Test checklist:**

- [ ] Tests fail when new logic is broken
- [ ] Happy path is covered
- [ ] Edge cases tested (empty input, invalid data)
- [ ] Use fixtures from `tests/conftest.py` for mocks
- [ ] No flaky tests (deterministic behavior)

### Configuration and secrets

- Store all configuration in `.env` (gitignored)
- Required: `OPENAI_API_KEY`
- Optional overrides: `CHAT_MODEL`, `VECTOR_STORE_DIR`, `LOG_LEVEL`
- Call `config.validate()` on startup to catch missing keys
- Never commit API keys or generated vector databases
- Use `config.get_logger()` for logging, respects `LOG_LEVEL`

## Additional resources

- **Documentation:** `README.md` for user guide and quick start
- **Design docs:** `docs/v0.2_design.md` for architecture decisions
- **Roadmap:** `docs/ROADMAP.md` for development plan and milestones
- **Project instructions:** `CLAUDE.md` for language and commit preferences
