# Architecture & Design Decisions

This document records the architectural decisions confirmed for the project, as well as unresolved decisions pending user confirmation.

---

## Confirmed Architectural Decisions

| ID | Topic | Decision | Rationale |
|---|---|---|---|
| **ADR-01** | Python Version | **Python 3.12** is mandatory. | Modern type hinting, native `tomllib`, performance improvements, and ecosystem stability. |
| **ADR-02** | Knowledge Store Backend | **TigerGraph** is the exclusive backend for both graph topology and vector indexes. | Hackathon requirement and core platform focus. Substitution with Qdrant, Neo4j, or Milvus is forbidden. |
| **ADR-03** | Retrieval Mechanism | **TigerGraph Hybrid Search** (vector similarity + graph topology traversal) is the preferred strategy. | Leverages native graph traversal combined with vector similarity in a unified database engine. |
| **ADR-04** | Specialist Tool Protocol | **Model Context Protocol (MCP)** for Layer 3 specialist tools. | Standardized tool exposition; enables pluggable tools and TigerGraph MCP server integration. |
| **ADR-05** | Agent Framework & Harness | **DeepAgents** investigated for Layer 5 policies; Layer 4 Harness retains strict control. | Explicit oversight of state, budgets, event logs, and evidence handling without framework lock-in. |
| **ADR-06** | Corpus Ground Truth | The local corpus is the **sole source of truth**. | Benchmark answers must be strictly grounded; pre-training priors and external web search are disabled for evaluation. |
| **ADR-07** | Reproducibility & Tracing | All runs must record deterministic hyperparameters and detailed step traces. | Research-grade benchmark comparing RAG, GraphRAG, and Agentic GraphRAG requires full auditability. |
| **ADR-08** | Application Structure | **Single Python package (`tgh`)** using `pyproject.toml`. | Avoids premature distributed microservices; simplifies reproducible experimentation. |

---

## Unresolved Decisions Pending Confirmation

1. **Virtual Environment & Package Tool**:
   - *Option A*: Standard Python `python3.12 -m venv` + `pip`.
   - *Option B*: Install and use `uv` for fast resolution and virtual environment management.
   - *Status*: Pending user confirmation (no software installed without prior approval).

2. **TigerGraph Deployment Target**:
   - *Option A*: Local Docker container running TigerGraph.
   - *Option B*: Remote TigerGraph Cloud instance.
   - *Option C*: Existing on-premise / standalone TigerGraph installation.
   - *Status*: Unresolved; connection parameters currently templated in `.env.example`.

3. **Corpus & Benchmark Dataset Selection**:
   - *Scope*: Domain and format for raw documents (`data/raw/`) and evaluation questions (`data/benchmarks/`).
   - *Status*: Unresolved; pending Phase 0B / Phase 1 planning.

4. **Embedding Model Selection**:
   - *Scope*: Model to generate vector embeddings for chunks and entities in TigerGraph.
   - *Status*: Unresolved; options include local embedding models (e.g. via fast sentence-transformers) or API-based embeddings.

5. **LLM Provider for Agentic Reasoning**:
   - *Scope*: Model provider and family to be orchestrated by Layer 4 / Layer 5.
   - *Status*: Unresolved; API endpoints and keys templated in `.env.example`.
