# Agentic GraphRAG on TigerGraph

### Adaptive reasoning over structured and unstructured Olympic data.
**TigerGraph × Native Vector Search × MCP × DeepAgents**

---

## 1. Project Overview

**Agentic GraphRAG on TigerGraph** is a high-precision knowledge retrieval and reasoning system built for the TigerGraph Hackathon. It compares three distinct knowledge retrieval and reasoning paradigms over an extensive corpus of Olympic documents:
1. **A0 (Standard Vector RAG)**: Baseline dense vector similarity search over document chunks.
2. **A1 (Adaptive GraphRAG)**: Relational expansion interleaving vector similarity seeds with graph neighborhood traversal.
3. **A4 (Agentic GraphRAG)**: Multi-step, agentic reasoning powered by **DeepAgents** and **SkillsMiddleware**, using bounded Model Context Protocol (**MCP**) specialist tools, governed by a deterministic **Execution Harness**, and backed by an authoritative **Evidence Ledger** with self-correcting verification.

All paradigms are benchmarked against the identical enterprise **TigerGraph** backend, identical chunk embeddings, and the same public benchmark question set.

---

## 2. Problem Statement

Standard semantic vector retrieval (Vector RAG) encounters structural failure modes in complex domain knowledge tasks:
- **Multi-Hop Relational Disconnect**: Vector embeddings capture semantic likeness but cannot reliably traverse entity relationships (e.g. associating an Olympic venue with an event date and retrieving the winning medalist).
- **Aggregations & Filtering**: Vector cosine distance cannot compute graph-wide statistics (e.g. counting participating nations, tracking chronological medal sequences).
- **Unverified Synthesis**: Generative models frequently hallucinate facts when operating over unverified retrieval chunks without explicit provenance and sufficiency checks.

---

## 3. Solution

Our solution unifies TigerGraph's graph database with native vector search under an agentic reasoning loop:
- **Unified Hybrid Knowledge Store**: Native 768-dimensional vector index residing inside TigerGraph alongside the complete knowledge graph topology (28,305 vertices, 57,737 edges).
- **DeepAgents + SkillsMiddleware**: Modular skill-based reasoning decomposed into discrete phases: Question Analysis, Strategy Selection, Graph Reasoning, Evidence Verification, and Answer Synthesis.
- **Authoritative Execution Harness**: A sandboxed environment that enforces budget constraints (`max_tool_calls = 15`, `max_repairs = 1`, `max_evidence_items = 50`), blocks arbitrary code/mutation, and maintains an immutable audit trail.
- **Evidence Ledger & Self-Repair**: A verifiable ledger of facts that validates answers before release, allowing a single bounded repair step to discover missing ground-truth facts.

---

## 4. System Architecture

The project is structured into six decoupled layers:

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
[Architecture diagram will be added here: docs/architecture.png]

---

## 5. Agentic Retrieval Strategies

- **A0 (Semantic RAG Baseline)**: Directly queries native TigerGraph vector search to retrieve top-k document chunks using cosine similarity.
- **A1 (Adaptive GraphRAG)**: Retrieves vector seeds, inspects incident relationships (`MENTIONS`, `HELD_AT`, `WON_BY`), and re-ranks documents based on combined semantic and topological connectivity.
- **A2 (Deterministic Graph Computation)**: Executes deterministic multi-hop traversals over TigerGraph vertices to answer composite questions (e.g. Venue $\to$ Event $\to$ Medalist) with zero hallucination.
- **A3 (Verification & Bounded Repair)**: Verifies whether the candidate answer is proven by entries in the Evidence Ledger. If evidence is insufficient, it triggers at most **one bounded repair cycle**.
- **A4 (Agentic GraphRAG Orchestration)**: An autonomous DeepAgents model-driven agent that analyzes incoming questions, selects between A0, A1, or A2, validates returned evidence, initiates verification, and generates grounded responses.

---

## 6. TigerGraph Knowledge Graph

TigerGraph serves as the exclusive and mandatory graph and vector database backend.

### Live Graph Topology Statistics
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

### Native Vector Search
- **Embedding Dimensions**: 768 dimensions
- **Metric**: Cosine similarity
- **Indexing**: HNSW vector index integrated natively in TigerGraph

### Graph Schema
[Live TigerGraph schema screenshot will be added here: docs/tigergraph_schema.png]

---

## 7. DeepAgents + SkillsMiddleware

The agent reasoning framework is built using the official **DeepAgents** standard with `SkillsMiddleware` loading five modular, discoverable skills:
1. `skills/question-analysis/SKILL.md`: Normalizes venue names, temporal cues (years, dates), and query intents.
2. `skills/retrieval-strategy-selection/SKILL.md`: Chooses the optimal retrieval path (A0, A1, or A2).
3. `skills/graph-reasoning/SKILL.md`: Orchestrates structured multi-hop graph exploration.
4. `skills/evidence-verification/SKILL.md`: Inspects the Evidence Ledger to verify candidate answers.
5. `skills/answer-synthesis/SKILL.md`: Synthesizes final grounded answers from verified facts.

---

## 8. Bounded MCP Specialist Tools

All agent interactions with TigerGraph are intermediated by allowlisted **Model Context Protocol (MCP)** specialist tools:
- **Allowlisted Tools**:
  - `tigergraph__get_node`: Inspect vertex attributes.
  - `tigergraph__get_edges`: Inspect outbound/inbound relationships.
- **Strict Security Boundaries**:
  - The agent has **no** direct access to TigerGraph credentials.
  - Schema mutation (`create_schema`, `drop_schema`) is strictly prohibited.
  - Arbitrary GSQL execution (`run_gsql`) and arbitrary Cypher (`run_cypher`) are blocked.
  - Raw shell, filesystem, or Python execution is forbidden.

---

## 9. Authoritative Execution Harness

The execution harness serves as the sandbox supervisor for all reasoning:
- **RunState & StateView**: RunState transitions are restricted to the `HarnessReducer`. Agents observe only a frozen, immutable `StateView`.
- **Budget Limits**:
  - `max_tool_calls = 15`
  - `max_repairs = 1` (a second repair attempt triggers immediate safe termination)
  - `max_evidence_items = 50`
- **Auditability**: Every tool call, skill execution, and state transition emits an immutable trace event for complete replay and analysis.

---

## 10. Evidence Ledger & A3 Verification

- **Evidence Ledger**: All retrieved chunks and graph facts are registered in an immutable ledger with assigned IDs (`ev-xxxx`), source types, confidence scores, and provenance paths.
- **A3 Verification**: Compares the candidate answer against the ledger. Answers lacking registered backing facts are marked `REPAIR_REQUIRED`.
- **Bounded Repair**: Triggers a single secondary retrieval to locate missing facts. In benchmark testing, this lifted gold-document recall to 100% in top-5/10/20.

---

## 11. Benchmark Results

Evaluated across the official 100-question public benchmark (`data/benchmarks/eval_public.jsonl`):

| Pipeline | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Mean Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0 Vector Baseline** | 68.00% | 83.00% | 86.00% | 90.00% | 0.7394 | 430.66 ms |
| **A1 Adaptive GraphRAG** | 66.00% | 81.00% | 86.00% | 90.00% | 0.7290 | 2373.00 ms |
| **A4 execution baseline** | **91.00%** | **100.00%** | **100.00%** | **100.00%** | **0.9445** | 7686.46 ms |

*Note: The A4 result is designated as the "A4 execution baseline". It demonstrates the effectiveness of agentic strategy routing, graph computation, and harness-governed verification over vector baselines.*

### Breakdown by Question Type (A4 Execution Baseline)
- **Temporal** (22 queries): Recall@1: **100.00%** | Recall@10: **100.00%** | MRR: **1.0000**
- **Multi-hop** (28 queries): Recall@1: **96.43%** | Recall@10: **100.00%** | MRR: **0.9732**
- **Lookup** (19 queries): Recall@1: **89.47%** | Recall@10: **100.00%** | MRR: **0.9474**
- **Aggregation** (21 queries): Recall@1: **85.71%** | Recall@10: **100.00%** | MRR: **0.9143**
- **Superlative** (10 queries): Recall@1: **70.00%** | Recall@10: **100.00%** | MRR: **0.8000**

---

## 12. Evaluation Methodology

The benchmark rigorously measures information retrieval effectiveness:
- **Recall@K**: Proportion of queries where at least one gold document is present in the top-K ranked documents.
- **MRR (Mean Reciprocal Rank)**: Evaluates the rank of the first relevant gold document ($1 / \text{rank}$).
- **Latency Profiling**: Millisecond-level end-to-end timing per query, capturing p50, p95, and p99 distributions.
- **Hidden Benchmark**: 50 hidden questions were executed for execution/latency evaluation. Gold labels/answers were not available, so no hidden accuracy claim is made.

---

## 13. Repository Structure

```
├── .env.example               # Template for required environment variables
├── .gitignore                 # Excludes secrets, caches, and hidden benchmark files
├── pyproject.toml             # Python 3.12 package configuration and dependencies
├── README.md                  # Comprehensive project documentation
├── docs/                      # Architectural reports and submission documentation
│   ├── SUBMISSION.md          # Full competition submission report
│   ├── GRAPHRAG_RETRIEVAL_DESIGN.md
│   └── STAGE_1_EXTRACTION_REPORT.md
├── experiments/
│   ├── results/
│   │   └── metrics_dashboard.md  # Detailed benchmark performance dashboard
│   └── runs/
│       └── agentic_graphrag_public_benchmark.json  # Frozen public benchmark results
├── skills/                    # DeepAgents SkillsMiddleware skills
│   ├── answer-synthesis/
│   ├── evidence-verification/
│   ├── graph-reasoning/
│   ├── question-analysis/
│   └── retrieval-strategy-selection/
├── src/tgh/                   # Core application source code
│   ├── embeddings/            # Nomic embedding wrapper & evaluation metrics
│   ├── evidence/              # Evidence Ledger implementation
│   ├── harness/               # Authoritative execution harness & state reducer
│   ├── ingestion/             # Document chunking and graph entity extraction
│   ├── mcp/                   # TigerGraph MCP client, contracts & specialist tools
│   ├── policies/              # Orchestrator & A3 verification/repair policies
│   ├── retrieval/             # A0 vector retriever & A1 GraphRAG retriever
│   └── telemetry/             # Execution event tracing
├── tests/                     # Unit and integration test suites
└── scripts/                   # Evaluation and benchmarking scripts
```

---

## 14. Setup

### Prerequisites
- Python 3.12+
- Access to a running TigerGraph instance with the `OlympicGraphRAG` graph loaded.

### Installation
```bash
# Clone the repository
git clone https://github.com/yeshiarK/tigergraph-hackathon.git
cd tigergraph-hackathon

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 15. Environment Variables

Create your local `.env` configuration from `.env.example`:
```bash
cp .env.example .env
```

Configure the following variables in `.env`:
```ini
TIGERGRAPH_HOST=https://your-tigergraph-instance.com
TIGERGRAPH_GRAPH_NAME=OlympicGraphRAG
TIGERGRAPH_SECRET=your_tigergraph_secret_here
GEMINI_API_KEY=your_gemini_api_key_here
```

*Note: Never commit `.env` or credentials to version control. The repository's `.gitignore` protects secrets automatically.*

---

## 16. Running the System

### Running Tests
Execute the focused unit and integration test suite:
```bash
pytest tests/ -v
```

### Running Code Quality Checks
```bash
ruff check src/ tests/
```

### Running the Public Benchmark
Execute the 100-question public evaluation:
```bash
python scripts/benchmark_agentic_graphrag.py
```

---

## 17. Reproducibility

Every component is deterministic and reproducible:
- **Fixed Embeddings**: Seed retrieval uses consistent 768-dim embeddings with frozen model weights.
- **Authoritative Reducer**: State transitions follow deterministic finite state machine rules.
- **Frozen Benchmark Artifact**: Full per-query ranks and retrieval logs are recorded in `experiments/runs/agentic_graphrag_public_benchmark.json`.

---

## 18. Limitations

1. **Superlative Attribute Reasoning**: Questions requiring comparative age calculations (e.g. oldest/youngest competitors) depend on the presence of structured birthdate vertex attributes.
2. **Network Traversal Latency**: Sequential multi-hop traversals over remote cloud instances incur WAN latency; batch reverse-edge traversal was implemented to mitigate this overhead.

---

## 19. Demo

To run a single live demonstration query tracing the complete A4 reasoning cycle:
```bash
python scripts/smoke_test_a3_a4.py
```
This demonstrates:
1. Question analysis and intent detection
2. Dynamic strategy selection
3. Bounded TigerGraph MCP tool execution
4. Evidence registration in the Evidence Ledger
5. Verification and grounded answer emission
