# TigerGraph Agentic GraphRAG Hackathon

A research-grade benchmark and evaluation platform comparing three distinct knowledge retrieval and reasoning paradigms:
1. **RAG** (Standard vector similarity retrieval + prompt synthesis)
2. **GraphRAG** (Graph-enhanced hybrid retrieval combining vector similarity and topological expansion + synthesis)
3. **Agentic GraphRAG** (Dynamic, multi-step agent reasoning, tool-mediated graph exploration, and budget-controlled evidence gathering)

All three approaches are evaluated against an identical underlying corpus and benchmark question suite to ensure rigorous, fair, and reproducible comparison.

---

## Architectural Constraint: TigerGraph Mandatory Backend

- **TigerGraph** is the **exclusive and mandatory** graph and vector database backend for this project.
- No substitution with Neo4j, Qdrant, Milvus, or other vector/graph stores is permitted.
- **TigerGraph's officially supported hybrid search** (interleaving vector similarity with GSQL graph traversal) is the preferred retrieval mechanism.
- Layer 3 exposes TigerGraph capabilities through the **Model Context Protocol (MCP)**.
- Benchmark answers must be strictly grounded in the ingested corpus. Pre-trained model parametric knowledge and external web search are strictly prohibited from answering benchmark questions.

---

## Six-Layer Architecture

The system is organized into six clean, decoupled architectural layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: Evaluation & Dashboard                             │
│ Benchmarking harness, comparative metrics, groundedness eval│
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Pipeline Policies                                  │
│ RAG  │  GraphRAG  │  Agentic GraphRAG (DeepAgents)          │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Agent Execution Harness                            │
│ Lifecycle, explicit state, budget limits, event traces      │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Specialists & MCP Tools                            │
│ TigerGraph MCP server, GSQL tools, vector search tools      │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Knowledge Stores                                   │
│ TigerGraph graph topology & native vector store             │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Ingestion                                          │
│ Corpus parsing, chunking, entity extraction, GSQL loading   │
└─────────────────────────────────────────────────────────────┘
```

1. **L1: Ingestion**: Document normalization, deterministic chunking, entity/relationship extraction, and TigerGraph loading jobs.
2. **L2: Knowledge Stores**: TigerGraph graph database (vertices, edges) and vector embeddings supporting hybrid search.
3. **L3: Specialists and MCP Tools**: MCP server interfaces and custom tool endpoints for GSQL queries, vector search, and subgraph extraction.
4. **L4: Agent Execution Harness**: Deterministic orchestration, state tracking, token/call budgets, execution limits, and event streaming.
5. **L5: Pipeline Policies**: Implementations of the three strategies being compared (RAG, GraphRAG, Agentic GraphRAG). DeepAgents is the target agent framework investigated here under strict harness controls.
6. **L6: Evaluation and Dashboard**: Automated benchmark execution, groundedness checks, precision/recall, latency/cost profiling, and comparative dashboards.

---

## Current Project Status: Phase 0A (Project Setup)

- **Current State**: Phase 0A (Setup only).
- **Available**: Repository layout, Git configuration, Python 3.12 project metadata (`pyproject.toml`), environment templates, testing configuration, and architectural documentation.
- **Unimplemented**: L1 Ingestion, L2 Knowledge Stores, L3 MCP Tools, L4 Harness, L5 Policies, and L6 Evaluation have not yet been implemented. No third-party application dependencies have been installed.
- For detailed setup and verification steps, see [docs/development/setup.md](file:///Users/yeshi/Desktop/tgh/docs/development/setup.md).
- For confirmed and unresolved design decisions, see [docs/development/decisions.md](file:///Users/yeshi/Desktop/tgh/docs/development/decisions.md).
- For live tracking of project tasks, see [docs/PROJECT_STATUS.md](file:///Users/yeshi/Desktop/tgh/docs/PROJECT_STATUS.md).
