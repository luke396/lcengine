# Repository Guidelines

## Project Structure & Module Organization

- `app/` hosts the RAG pipeline modules: configuration, pipeline orchestration, conversation management, document processing, and embedding services.
- `app/vector_store/` defines the lightweight vector index adapters; runtime stores default to `data/`.
- `app.py` exposes the Streamlit UI and is what `main.py` launches through `python main.py`.
- `tests/` mirrors runtime modules with fast unit coverage plus integration and performance suites.
- `data/` is the scratch space for uploads and cached vectors; keep large artifacts out of git.

## Build, Test, and Development Commands

- `uv sync` — install or update dependencies from `pyproject.toml`/`uv.lock`.
- `python main.py` — launch the Streamlit app on port 8501 with logging configured.
- `streamlit run app.py --server.port 8501` — direct UI run when debugging layout changes.
- `pytest` — execute the suite; combine with `-m "not slow"` to skip performance checks.
- `pytest --cov=app --cov-report=term-missing` — confirm coverage before submitting.
- `ruff check app tests` / `ruff format app tests` — lint and format; resolve issues before pushing.
- `pyright app tests` — static type checks using the repo’s `standard` mode.

## Coding Style & Naming Conventions

Use four-space indentation, PEP 8 layout, and type hints on public APIs. Keep modules and functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`. Ruff enforces Google-style docstrings and most formatting rules; run it locally so CI passes. Prefer small helpers and log through `config.get_logger` so messages respect global settings.

## Testing Guidelines

Place new tests in `tests/` using the `test_<subject>.py` pattern. Reuse fixtures from `tests/conftest.py` and mark long scenarios with `@pytest.mark.slow`. Assertions should inspect retrieved context, dialogue state, and edge cases. Keep coverage on new paths close to existing levels (≈85%); explain any exceptions inside the PR.

## Commit & Pull Request Guidelines

Commits follow Conventional Commit prefixes (`feat:`, `fix:`, `test:`, `chore:`) with concise scopes. For pull requests include a summary, linked issue or roadmap item, test command output, configuration updates (e.g., new env vars), and visuals or transcripts when the UI changes.

## Configuration & Secrets

Create a `.env` (gitignored) with `OPENAI_API_KEY`; override `CHAT_MODEL`, `VECTOR_STORE_DIR`, or log levels only as needed. The app calls `config.validate()` on startup, so verify keys locally before committing. Never commit API keys or generated vector databases; share sanitized setup notes when new values are needed.

## Code Quality & Static Analysis

Run `ruff` and `pyright` locally to catch issues. Address all linting errors and type warnings unless there’s a documented exception.

Keep `pyproject.toml` and `uv.lock` in sync with dependency changes; run `uv sync` after modifying either.
