# TigerGraph Agentic GraphRAG Hackathon Submission

**Project Title**: Agentic GraphRAG on TigerGraph  
**Subtitle**: Adaptive reasoning over structured and unstructured Olympic data using TigerGraph, Native Vector Search, Model Context Protocol (MCP), and DeepAgents.  
**Repository**: [https://github.com/yeshiarK/tigergraph-hackathon](https://github.com/yeshiarK/tigergraph-hackathon)

---

## 1. Project Overview

This project presents a production-grade, research-rigorous benchmark and implementation comparing three distinct knowledge retrieval and reasoning paradigms:
1. **A0 (Standard Vector RAG)**: Baseline semantic similarity retrieval over dense document chunk embeddings.
2. **A1 (Adaptive GraphRAG)**: Hybrid relational retrieval interleaving vector seed retrieval with 1-to-2-hop graph topological expansion.
3. **A4 (Agentic GraphRAG)**: Dynamic multi-step reasoning orchestrated with DeepAgents and bounded TigerGraph MCP specialist tools, governed by an authoritative Execution Harness and verified via an Evidence Ledger.

All implementations operate over the exact same underlying corpus of Olympic documents and the same live **TigerGraph** enterprise database instance.

---

## 2. Problem Statement

Standard Retrieval-Augmented Generation (Vector RAG) struggles with complex domain knowledge for three reasons:
- **Relational Disconnect**: Vector embeddings capture semantic proximity but fail to follow multi-hop relational dependencies (e.g. connecting a specific venue name to an event date and then identifying the medalist).
- **Aggregations and Temporal Constraints**: Vector similarity cannot perform deterministic filtering or topological counts across entities (e.g. counting participating nations or filtering by Olympic edition year).
- **Hallucination Without Verification**: Unbounded LLM retrieval often returns plausible-sounding text chunks with no guarantee of factual grounding, with no ledger tracking what evidence supports what claim.

---

## 3. System Architecture

The system is constructed around six decoupled architectural layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Layer 6: Evaluation & Metrics Dashboard                                │
│ Automated benchmarking, MRR / Recall@K, latency, and groundedness      │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 5: Reasoning Policies & Agent Layer                              │
│ A0 Vector RAG │ A1 Adaptive GraphRAG │ A2 Graph Tools │ A3 Verifier    │
│ A4 DeepAgents Orchestration (SkillsMiddleware)                         │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 4: Authoritative Execution Harness                               │
│ RunState (immutable view), HarnessReducer, ExecutionBudget, Log, Ledger│
├────────────────────────────────────────────────────────────────────────┤
│ Layer 3: Specialists & MCP Tools                                       │
│ Bounded TigerGraph MCP tools (tigergraph__get_node, get_edges)         │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Knowledge Stores (TigerGraph)                                 │
│ Graph topology (28,305 vertices, 57,737 edges) & native vector search │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Ingestion & Normalization                                     │
│ Document parsing, chunking, NER/REL extraction, GSQL batch loading     │
└────────────────────────────────────────────────────────────────────────┘
```

### Architecture Diagram
<!-- Placeholder for Architecture Diagram -->
[Architecture diagram will be added here: docs/architecture.png]

---

## 4. Retrieval Strategies: A0, A1, A2, A3, A4

- **A0 (Semantic Vector RAG)**: Queries dense embeddings (768-dim) using TigerGraph native cosine distance to retrieve top-k chunks.
- **A1 (Adaptive GraphRAG)**: Uses vector seeds as entry points, traverses relational edges (`MENTIONS`, `HELD_AT`, `WON_BY`), and re-ranks candidate documents using seed similarity and graph connectivity scores.
- **A2 (Deterministic Graph Computation)**: Executes specialized composite traversals directly on graph topology to answer structured queries (e.g., Venue $\to$ `reverse_HELD_AT` $\to$ Events $\to$ `WON_BY` $\to$ Athlete) with zero hallucination risk.
- **A3 (Evidence Verification & Bounded Repair)**: Verifies that candidate claims are directly supported by registered facts in the Evidence Ledger. If unverified, it triggers at most **one bounded repair cycle** to discover missing evidence.
- **A4 (Agentic GraphRAG Orchestration)**: Dynamic orchestrator powered by DeepAgents and SkillsMiddleware. The agent analyzes the input query, selects the optimal strategy (A0, A1, or A2), gathers structured evidence, verifies factual sufficiency, and synthesizes a grounded answer.

---

## 5. TigerGraph Knowledge Graph

TigerGraph serves as the exclusive and mandatory graph and vector storage backend.

### Live Graph Statistics
- **Documents**: 2,951
- **Document Chunks**: 9,348
- **Extracted Entities**: 8,003
  - **Events**: 2,187
  - **People / Athletes**: 4,333
  - **Teams**: 985
  - **Countries**: 136
  - **Sports**: 42
  - **Venues**: 320
- **Total Vertices**: **28,305**
- **Total Edges**: **57,737**

### Vector Search Configuration
- **Dimensions**: 768-dimensional embeddings
- **Similarity Metric**: Cosine similarity
- **Index Type**: HNSW vector index
- **Storage**: Native TigerGraph vector storage integrated with vertex attributes

### Schema Visualization
<!-- Placeholder for Live TigerGraph Schema Screenshot -->
[Live TigerGraph schema screenshot will be added here: docs/tigergraph_schema.png]

---

## 6. DeepAgents + SkillsMiddleware & Bounded MCP

The agent layer is implemented using **DeepAgents** with the `SkillsMiddleware` pattern:
1. `skills/question-analysis`: Identifies venues, dates, years, and question intents.
2. `skills/retrieval-strategy-selection`: Routes queries to A0, A1, or A2.
3. `skills/graph-reasoning`: Guides multi-hop graph exploration.
4. `skills/evidence-verification`: Inspects the Evidence Ledger for sufficient backing facts.
5. `skills/answer-synthesis`: Formulates grounded responses from verified evidence.

### Strict MCP Security & Tool Boundaries
The system interfaces with TigerGraph through allowlisted Model Context Protocol (MCP) specialist tools.
- **Allowlisted**: `tigergraph__get_node`, `tigergraph__get_edges`.
- **Strictly Blocked**: Schema mutation (`create_schema`, `drop_schema`), arbitrary GSQL (`run_gsql`), arbitrary Cypher (`run_cypher`), and shell/file modifications.
- The model never possesses direct database credentials or arbitrary execution capabilities.

---

## 7. Execution Harness & Evidence Ledger

The Execution Harness provides deterministic guardrails over agent operations:
- **RunState**: Authoritative execution state managed solely by `HarnessReducer`. Outside callers and agent models receive an immutable `StateView`.
- **Budgets Enforced**:
  - `max_tool_calls = 15`
  - `max_repairs = 1` (a second repair attempt triggers immediate safe termination)
  - `max_evidence_items = 50`
- **Evidence Ledger**: Immutable register where all discovered chunks and graph facts are assigned unique identifiers (`ev-xxxx`), confidence scores, document provenance, and timestamped audit logs.

---

## 8. Benchmark Results

Evaluated across the 100-question public benchmark (`data/benchmarks/eval_public.jsonl`):

| Pipeline | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Mean Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0 Vector Baseline** | 68.00% | 83.00% | 86.00% | 90.00% | 0.7394 | 430.66 ms |
| **A1 Adaptive GraphRAG** | 66.00% | 81.00% | 86.00% | 90.00% | 0.7290 | 2373.00 ms |
| **A4 execution baseline** | **91.00%** | **100.00%** | **100.00%** | **100.00%** | **0.9445** | 7686.46 ms |

### Results by Question Type (A4 Execution Baseline)
- **Temporal** (22 queries): Recall@1: **100.00%** | Recall@10: **100.00%** | MRR: **1.0000**
- **Multi-hop** (28 queries): Recall@1: **96.43%** | Recall@10: **100.00%** | MRR: **0.9732**
- **Lookup** (19 queries): Recall@1: **89.47%** | Recall@10: **100.00%** | MRR: **0.9474**
- **Aggregation** (21 queries): Recall@1: **85.71%** | Recall@10: **100.00%** | MRR: **0.9143**
- **Superlative** (10 queries): Recall@1: **70.00%** | Recall@10: **100.00%** | MRR: **0.8000**

---

## 9. Hidden Evaluation Methodology

50 hidden questions were executed for execution/latency evaluation. Gold labels/answers were not available, so no hidden accuracy claim is made. All benchmarks and evaluations adhere to zero-leakage protocols.

---

## 10. Reproducibility & Setup

### Requirements
- Python >= 3.12
- Active TigerGraph instance with OlympicGraphRAG schema and vector index

### Setup Steps
```bash
# 1. Clone repository
git clone https://github.com/yeshiarK/tigergraph-hackathon.git
cd tigergraph-hackathon

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 3. Configure environment
cp .env.example .env
# Edit .env with your TigerGraph credentials and GEMINI_API_KEY
```

### Running Tests
```bash
pytest tests/ -v
ruff check src/ tests/
```

### Running the Public Benchmark
```bash
python scripts/benchmark_agentic_graphrag.py
```

---

## 11. Limitations

1. **Superlative Complexities**: Superlative queries (e.g. oldest/youngest medalist) occasionally suffer if biographical attributes are stored solely in unstructured text rather than structured vertex properties.
2. **WAN Traversal Latency**: Multi-hop edge expansions across remote TigerGraph instances involve HTTP round trips; batch traversal optimization and connection pooling are used to mitigate latency.

---

## 12. Demo Instructions

To run an interactive single-query demonstration of A4 Agentic GraphRAG:
```bash
python scripts/smoke_test_a3_a4.py
```
This demonstrates question analysis, strategy selection, TigerGraph traversal, evidence registration in the ledger, and verification.
