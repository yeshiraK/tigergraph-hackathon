# System Architecture

## Overview

The TigerGraph Agentic GraphRAG project provides an empirical research platform to compare three information retrieval and synthesis paradigms over an identical corpus:
1. **Standard RAG**
2. **Graph-Enhanced RAG (GraphRAG)**
3. **Agentic GraphRAG**

The system is designed as a single Python application (avoiding premature microservices) with clean separation of concerns across six conceptual layers.

---

## Architectural Layers

### Layer 1: Ingestion
- **Objective**: Ingest raw source documents from `data/raw/`, normalize text, generate deterministic chunks, extract entities and relationships, produce vector embeddings, and load all assets into TigerGraph.
- **Components (Planned)**:
  - Document parser & text cleaner.
  - Chunking engine (preserving document offset & hierarchy).
  - Entity/Relation extraction pipeline.
  - TigerGraph GSQL loading job coordinator.
- **Status**: Not implemented in Phase 0A.

### Layer 2: Knowledge Stores
- **Objective**: Act as the single source of truth for all graph topology and vector indexes.
- **Components**:
  - TigerGraph instance housing vertices (Documents, Chunks, Entities, Concepts) and edges.
  - TigerGraph native vector search index for chunk and entity embeddings.
  - Pre-installed GSQL queries for hybrid retrieval and k-hop neighborhood expansion.
- **Status**: Schema and query directories initialized in `graph/`; connection and schema deployment pending subsequent phases.

### Layer 3: Specialists and MCP Tools
- **Objective**: Expose specialized capabilities to reasoning layers through the Model Context Protocol (MCP).
- **Components (Planned)**:
  - TigerGraph MCP server client / adapter.
  - Specialist tools:
    - `vector_search(query_vector, top_k)`
    - `graph_neighborhood(vertex_id, depth, edge_types)`
    - `hybrid_retrieve(query_text, query_vector, filters)`
    - `get_evidence_snippet(chunk_id)`
- **Status**: Not implemented in Phase 0A.

### Layer 4: Agent Execution Harness
- **Objective**: Orchestrate agent execution loops with deterministic controls, protecting against runaway loops and non-reproducible behavior.
- **Components (Planned)**:
  - Loop controller with explicit budget thresholds (maximum steps, max token budget, timeout).
  - Explicit state manager (tracking working memory, active hypothesis, collected evidence references).
  - Event stream generator emitting detailed audit events to `src/tgh/telemetry/`.
  - Refusal and groundedness guardrails.
- **Status**: Not implemented in Phase 0A.

### Layer 5: Pipeline Policies
- **Objective**: Encapsulate the three competing retrieval-generation paradigms.
- **Components (Planned)**:
  - `RAGPolicy`: One-shot vector search followed by answer generation.
  - `GraphRAGPolicy`: Hybrid vector + graph expansion query followed by structured context synthesis.
  - `AgenticGraphRAGPolicy`: Multi-step reasoning policy (investigating DeepAgents) invoking Layer 3 MCP tools to dynamically navigate TigerGraph.
- **Status**: Not implemented in Phase 0A.

### Layer 6: Evaluation and Dashboard
- **Objective**: Execute reproducible benchmarks across all policies, evaluate groundedness, and generate comparative scorecards.
- **Components (Planned)**:
  - Benchmark test runner iterating over identical question-answer pairs in `data/benchmarks/`.
  - Groundedness checker: ensures all assertions in answers map to cited evidence from the corpus.
  - Metric evaluators: Factual accuracy, Hallucination/Unsupported rate, Latency, Token/Cost efficiency.
  - Dashboard report generator.
- **Status**: Not implemented in Phase 0A.

---

## Core Guarantees & Constraints

1. **TigerGraph Exclusivity**: All graph and vector data operations run on TigerGraph. No secondary databases are permitted.
2. **Corpus Ground Truth**: The corpus in `data/` is the sole source of truth. Models must not rely on pre-training priors or external web searches for benchmark evaluations.
3. **Execution Reproducibility**: Seed control, deterministic prompts, logged hyperparameters, and full telemetry traces allow exact replication of experiments.
