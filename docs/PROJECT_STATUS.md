# Project Status Record

## Current Phase
**Phase 0A: Project Setup**

---

## Verified Environment Details

- **Project Root**: `/Users/yeshi/Desktop/tgh`
- **Initial State**: Empty directory (verified via directory listing)
- **Python Runtime**: `Python 3.12.0` (verified at `/usr/local/bin/python3.12` and `/Users/yeshi/.pyenv/shims/python3`)
- **Python venv**: Functional (`python3.12 -m venv` verified)
- **Git Version**: `git version 2.48.1` (verified)
- **Git Repository**: Initialized with default branch `main`

---

## Completed Tasks (Phase 0A)

- [x] **Workspace Inspection**: Confirmed current working directory, verified folder was initially empty, and confirmed Python 3.12 and Git availability.
- [x] **Git Repository Initialization**: Initialized repository, set branch to `main`, and committed no files prematurely.
- [x] **Gitignore Configuration**: Created `.gitignore` protecting datasets (`data/raw`, `data/processed`, `data/benchmarks`), environment files (`.env*`), caches (`.pytest_cache`, `__pycache__`), TigerGraph index/export files, and experiment run logs.
- [x] **Environment Template**: Created `.env.example` documenting TigerGraph connection settings, MCP server settings, and LLM configuration without committing secrets.
- [x] **Packaging Metadata**: Created `pyproject.toml` targeting Python `>=3.12` using PEP 621 metadata, setuptools backend, and dev dependencies (`pytest`, `ruff`).
- [x] **Directory Structure**: Created six-layer architecture structure (`config`, `data`, `graph`, `src/tgh/`, `tests`, `scripts`, `experiments`, `docs`) with domain, ingestion, mcp, harness, policies, evidence, evaluation, and telemetry modules.
- [x] **Initial Project Documentation**: Created `README.md`, `docs/architecture/architecture.md`, `docs/development/setup.md`, and `docs/development/decisions.md`.
- [x] **Project Quality Configuration**: Created minimal configuration test in `tests/test_project_config.py` verifying Python 3.12 runtime and packaging metadata using Python standard library.
- [x] **Project Status Tracking**: Created `docs/PROJECT_STATUS.md`.

---

## Pending Tasks

### Phase 0B / Immediate Next Actions
- [ ] User decision on virtual environment tool (`python3.12 -m venv` vs `uv`).
- [ ] Virtual environment creation and dev dependency installation (`pytest`, `ruff`).
- [ ] Specification of TigerGraph instance target (local Docker vs cloud vs on-prem).
- [ ] Corpus dataset selection and benchmark question format specification.

### Unimplemented Architectural Layers (Future Phases)
- [ ] **L1 Ingestion**: Document parser, chunking engine, entity extraction, GSQL loading jobs.
- [ ] **L2 Knowledge Stores**: TigerGraph GSQL schema deployment and vector indexing.
- [ ] **L3 Specialists & MCP**: TigerGraph MCP server integration and retrieval tools.
- [ ] **L4 Execution Harness**: Loop controls, budget tracking, state management, telemetry streaming.
- [ ] **L5 Pipeline Policies**: RAG baseline, GraphRAG pipeline, Agentic GraphRAG (DeepAgents).
- [ ] **L6 Evaluation**: Benchmark test suite, groundedness verification, comparison dashboard.

---

## Decisions Requiring Confirmation

1. **Virtual Environment Setup**: Confirm whether to create a `.venv` with `python3.12 -m venv .venv` and install dev dependencies, or if another package manager is desired.
2. **TigerGraph Connectivity**: Confirm the target TigerGraph endpoint details (local Docker container vs remote instance).
3. **Corpus Selection**: Confirm the source corpus domain to be ingested for the benchmark.
4. **Model & Embedding Providers**: Confirm the preferred LLM and embedding model endpoints.
