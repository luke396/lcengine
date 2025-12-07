# LCEngine

A production-grade Retrieval-Augmented Generation (RAG) system with agent capabilities, designed as a learning copilot for deep learning and LLM development.

[![CI](https://github.com/luke396/lcengine/workflows/CI/badge.svg)](https://github.com/luke396/lcengine/actions)
[![codecov](https://codecov.io/gh/luke396/lcengine/branch/main/graph/badge.svg)](https://codecov.io/gh/luke396/lcengine)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

LCEngine is a native RAG implementation that demonstrates deep understanding of retrieval, embedding, and generation pipelines. Built from the ground up without heavy framework dependencies, it showcases both technical depth and production engineering practices.

**Key Differentiators:**

- **Native Implementation**: Core RAG logic built from scratch for maximum transparency and control
- **Long-term Memory**: Innovative note-taking and mistake-tracking system with vector-based recall
- **Evaluation-Driven**: Built-in RAGAS metrics and continuous quality assessment
- **Production-Ready**: Comprehensive testing (85%+ coverage), CI/CD, and monitoring capabilities

## Features

### Core Capabilities

- **Multi-Source Knowledge Ingestion**

  - Local document upload (PDF, TXT, Markdown)
  - URL and GitHub repository ingestion (planned v0.3)
  - Web search agent with domain whitelisting (planned v0.3)

- **Advanced Retrieval** (v0.4)

  - Hybrid search: BM25 (sparse) + vector (dense) retrieval
  - Cross-encoder reranking for precision
  - Metadata-aware filtering and weighting
  - Time-decay and source-based ranking

- **Long-term Memory System** (v0.2)

  - Save valuable Q&A as notes
  - Track mistakes to avoid repetition
  - Metadata tagging (topic, source, timestamp)
  - Weighted retrieval bias for personalized recall

- **Conversation Management**

  - Multi-turn dialogue with context maintenance
  - Standalone query generation for follow-up questions
  - Configurable conversation history depth

- **Quality Assurance**
  - RAGAS evaluation framework (Faithfulness, Context Recall, Answer Relevancy)
  - Hit@k metrics and P95 latency tracking
  - Cost monitoring and API usage analytics
  - Automated baseline comparison

### Planned Features

- **v0.3**: URL/GitHub ingestion, search agent with confidence-based triggering
- **v0.4**: Hybrid retrieval, cross-encoder reranking, metadata filtering
- **v0.5**: Learning modes (explanation + study plan + practice), problem-solving workflows
- **v0.6**: Production monitoring (LangSmith/LangFuse), user feedback loop

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lcengine.git
cd lcengine

# Install dependencies with uv
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running the Application

```bash
# Launch Streamlit UI (recommended)
python main.py

# Or run Streamlit directly for debugging
streamlit run app.py --server.port 8501
```

The application will be available at `http://localhost:8501`.

### Basic Usage

1. **Upload Documents**: Use the sidebar to upload PDF or TXT files
2. **Ask Questions**: Type your question in the chat interface
3. **Save Knowledge**: Click "Save as Note" to add valuable Q&A to long-term memory
4. **Review Sources**: Check the debug panel for retrieved chunks and similarity scores

## Architecture

### Technology Stack

| Layer            | Technology                             | Rationale                                     |
| ---------------- | -------------------------------------- | --------------------------------------------- |
| **Embedding**    | OpenAI text-embedding-3-small          | High quality, cost-effective                  |
| **Vector Store** | FAISS + SQLite                         | Fast similarity search + metadata persistence |
| **LLM**          | OpenAI gpt-4.1-nano-2025-04-14         | Low latency, good quality/cost ratio          |
| **Retrieval**    | Hybrid (BM25 + Vector) + Cross-Encoder | Balance keyword and semantic search           |
| **UI**           | Streamlit                              | Rapid prototyping, sufficient for demos       |
| **Testing**      | Pytest + Pre-commit                    | Quality assurance automation                  |

### Project Structure

```
lcengine/
├── app/                      # Core application modules
│   ├── config.py            # Configuration management
│   ├── models.py            # Data models (DocumentChunk, ConversationTurn)
│   ├── embeddings.py        # OpenAI embeddings service
│   ├── document_processing.py  # PDF/TXT loading and chunking
│   ├── vector_store/        # Vector storage implementations
│   │   ├── sqlite_store.py  # SQLite + NumPy (v0.1)
│   │   └── faiss_store.py   # FAISS integration (v0.2, planned)
│   ├── pipeline.py          # RAG pipeline orchestration
│   └── conversation.py      # Multi-turn dialogue management
├── tests/                   # Comprehensive test suite
│   ├── conftest.py          # Pytest fixtures and mocks
│   ├── test_*.py            # Unit and integration tests
│   └── data/                # Test datasets
├── docs/                    # Documentation
│   ├── v0.2_design.md       # Architecture decisions
│   ├── DEVLOG.md            # Development journal (planned)
│   └── experiments.md       # Evaluation results (planned)
├── app.py                   # Streamlit UI entry point
├── main.py                  # CLI launcher
└── pyproject.toml           # Project metadata and dependencies
```

### Data Flow

```
Document Upload
    ↓
Text Extraction & Chunking
    ↓
Embedding Generation (OpenAI API)
    ↓
Vector Storage (FAISS/SQLite)
    ↓
User Query → Standalone Query Generation
    ↓
Similarity Search (Cosine/L2)
    ↓
Context Building (Top-k chunks)
    ↓
LLM Generation (OpenAI Chat API)
    ↓
Response + Source Attribution
```

## Development

### Setup Development Environment

```bash
# Install dependencies with dev tools
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit checks manually
uv run pre-commit run --all-files --show-diff-on-failure
```

### Running Tests

```bash
# Run all tests except slow ones
uv run pytest -m "not slow"

# Run all tests including performance benchmarks
uv run pytest

# Run with coverage report
uv run pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Lint and format with ruff
ruff check app tests
ruff format app tests

# Type checking with pyright
pyright app tests
```

### Configuration

Environment variables (`.env` file):

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional - API Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
API_USER_AGENT=LCEngine/1.0

# Optional - Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4.1-nano-2025-04-14
CHAT_MAX_TOKENS=500
CHAT_TEMPERATURE=0.7

# Optional - Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optional - Storage Configuration
VECTOR_BACKEND=faiss  # or sqlite
VECTOR_STORE_DB_PATH=data/vector_store.db
VECTOR_STORE_DIR=data/vectors
FAISS_INDEX_PATH=data/faiss/index.faiss

# Optional - Logging
LOG_LEVEL=INFO
OPENAI_LOG_LEVEL=WARNING
ENVIRONMENT=development
```

## Evaluation

Run the evaluation script to assess retrieval and generation quality:

```bash
# Full evaluation (v0.2+)
python evaluate.py \
    --dataset tests/data/evaluation_dataset.json \
    --k 5 \
    --vector-backend faiss \
    --output data/eval_runs/$(date +%Y%m%d_%H%M%S).json

# Dry run (retrieval only, no LLM calls)
python evaluate.py --dataset tests/data/evaluation_dataset.json --dry-run

# Limited subset for quick validation
python evaluate.py --dataset tests/data/evaluation_dataset.json --limit 10
```

### Metrics Tracked

- **Hit@k**: Percentage of queries where correct document is in top-k results
- **RAGAS Faithfulness**: Measures hallucination (answer grounded in context)
- **RAGAS Context Precision/Recall**: Retrieval quality metrics
- **RAGAS Answer Relevancy**: How well answer addresses the question
- **P95 Latency**: 95th percentile response time
- **Cost per Query**: OpenAI API usage cost

Results are saved to `data/eval_runs/` with timestamps for version comparison.

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Linting**: Ruff format/check, trailing whitespace removal
- **Type Checking**: Pyright static analysis
- **Fast Tests**: Run on every PR (excludes `@pytest.mark.slow`)
- **Slow Tests**: Run on main branch and manual dispatch (performance benchmarks)

Pre-commit hooks ensure code quality before commits:

- Ruff formatting and linting
- Pyright type checking
- Pytest fast tests
- File cleanup (trailing whitespace, EOF newlines)

## Design Decisions

### Why Native Implementation?

**Advantages:**

- **Transparency**: Every line of retrieval and generation logic is visible and understandable
- **Control**: Fine-grained control over chunking, embedding, and ranking strategies
- **Performance**: No framework overhead, optimized for specific use case
- **Learning**: Demonstrates deep understanding of RAG fundamentals

**Trade-offs:**

- Slower initial development compared to frameworks (LangChain, LlamaIndex)
- Need to implement common utilities (chunking, ranking) from scratch
- Less community support for edge cases

**When Frameworks Are Used:**

- Data ingestion (v0.3): LlamaIndex Readers for URL/GitHub (efficiency gain)
- Complex orchestration (v0.5): LangGraph for multi-step workflows (state management complexity)
- Monitoring (v0.6): LangSmith/LangFuse for tracing (mature tooling)

**Principle**: Build core algorithms natively, use frameworks for peripheral functionality.

## Roadmap

**Public Milestones:**

- ✅ **v0.1** (Completed): Basic RAG with NumPy vector store
- 🚧 **v0.2** (In Progress): FAISS upgrade, long-term memory, evaluation framework
- 📅 **v0.3** (Planned): Multi-source ingestion, search agent
- 📅 **v0.4** (Planned): Hybrid retrieval, cross-encoder reranking
- 📅 **v0.5** (Planned): Learning modes, LangGraph workflows
- 📅 **v0.6** (Planned): Production monitoring, data feedback loop

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation framework
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [Streamlit](https://streamlit.io/) for rapid UI development
- OpenAI for embedding and LLM APIs

---

**Note**: This project is designed as a learning tool and interview portfolio piece. It demonstrates production engineering practices (testing, CI/CD, evaluation) while maintaining code transparency through native implementation of core algorithms.
