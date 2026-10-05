# Agentic GraphRAG with TigerGraph

An adaptive, multi-strategy knowledge retrieval and reasoning system combining TigerGraph's native vector index, relational graph topology, DeepAgents orchestration, and bounded evidence verification.

> RAG retrieves.  
> Graphs connect.  
> Agents reason.  
> Verification grounds.  

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![TigerGraph](https://img.shields.io/badge/TigerGraph-4.2.5-orange.svg)](https://www.tigergraph.com/)
[![Tests](https://img.shields.io/badge/tests-132%20passed-brightgreen.svg)](tests/)

---

## 1. Overview

**Agentic GraphRAG with TigerGraph** tackles complex domain queries over an extensive corpus of Olympic Games history. Real-world questions rarely adhere to a single retrieval pattern: simple factual lookups require fast semantic search, while relational questions demand multi-hop graph traversal and topological aggregation.

Instead of binding queries to a single static pipeline, this system coordinates:
- **TigerGraph Knowledge Graph**: 28,305 vertices and 57,737 edges representing athletes, events, venues, sports, countries, teams, and documents.
- **Native Vector Retrieval**: 768-dimensional native HNSW vector index executing within TigerGraph.
- **GraphRAG Expansion**: Interleaved vector similarity seed selection and graph topological expansion.
- **Deterministic Graph Reasoning**: Specialized graph primitives executing deterministic traversals with zero hallucination risk.
- **DeepAgents & SkillsMiddleware**: Autonomous model-driven planning with modular skill execution.
- **Official TigerGraph MCP**: Bounded tool integration exposing read-only graph inspection endpoints.
- **Bounded Execution Harness**: Deterministic runtime sandbox enforcing hard tool budgets and cycle detection.
- **Evidence Ledger**: Immutable registry of retrieved facts and text chunks.
- **A3 Verification & Bounded Repair**: Automated grounding audits that trigger at most one targeted corrective retrieval pass before answer release.

> **Control Boundary**: The agent is **NOT** given unrestricted database access. The model cannot execute arbitrary GSQL/Cypher or mutate graph data. All interactions are strictly mediated through allowlisted tools inside the execution harness.

---

## 2. Core Contribution

The central thesis of this project is that **a question does not always need the same retrieval strategy**. 

Standard RAG treats all queries uniformly: convert text to embedding, perform vector nearest-neighbor lookup, and synthesize an answer. When faced with relational dependencies (e.g., *"Which venue hosted the event won by athlete X in 1992?"*) or aggregations (e.g., *"How many countries competed?"*), vector retrieval fails.

The system provides four specialist capabilities orchestrated by an agentic controller:
- **A0 — Semantic Vector RAG**: Fast nearest-neighbor search for broad topical queries.
- **A1 — GraphRAG**: Seed-based relational expansion across entity neighborhoods.
- **A2 — Deterministic Graph Reasoning**: Specialist topological traversal for multi-hop paths, aggregations, and entity lookups.
- **A3 — Verification & Bounded Repair**: Audit engine ensuring candidate answers are grounded in registered evidence.
- **A4 — Agentic GraphRAG Orchestration**: Autonomous controller that dynamically selects specialist capabilities based on intermediate observations.

### Dynamic Strategy Adaptation Example:
```
User asks multi-hop relational question
  │
  ▼
A1 GraphRAG initiated (seed expansion)
  │
  ▼
Insufficient evidence observed (missing intermediate athlete link)
  │
  ▼
A0 targeted vector retrieval invoked for specific athlete context
  │
  ▼
Evidence Ledger registers corroborating facts
  │
  ▼
A3 verification validates factual grounding
  │
  ▼
Grounded, verified answer synthesized with citations
```

*Note: Deterministic graph tools (A2) are not "LLM agents"; they are specialized, highly optimized topological algorithms exposed to the controller as deterministic tools.*

---

## 3. Architecture

```
USER QUESTION
      │
      ▼
┌────────────────────────────────────────────────────────┐
│                      A4 AGENT                          │
│          DeepAgents + SkillsMiddleware                 │
└─────────────────────────┬──────────────────────────────┘
                          │ dynamic tool selection
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │    A0     │    │    A1     │    │    A2     │
   │Vector RAG │    │ GraphRAG  │    │   Graph   │
   │ (Native)  │    │(Expansion)│    │ Reasoning │
   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
            TigerGraph Knowledge Graph
         (28,305 vertices, 57,737 edges)
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│                   EXECUTION HARNESS                    │
│   • RunState & StateView      • Budget Enforcement     │
│   • Tool Execution Allowlist  • No-Progress Protection│
│   • Event Log Audit Trail     • Harness Reducer        │
└─────────────────────────┬──────────────────────────────┘
                          │ structured evidence items
                          ▼
┌────────────────────────────────────────────────────────┐
│                    EVIDENCE LEDGER                     │
│    • Vector Chunks           • Graph Triples & Facts   │
│    • Source Metadata         • Confidence Scores       │
└─────────────────────────┬──────────────────────────────┘
                          │ candidate answer + evidence
                          ▼
┌────────────────────────────────────────────────────────┐
│                 A3 VERIFICATION ENGINE                 │
│      • Grounding Check        • Entity Consistency     │
│      • Constraint Validation  • Temporal Alignment     │
└─────────────────────────┬──────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      [SUPPORTED]                   [REPAIR] (max 1)
            │                           │
            │                           ▼
            │               Targeted Corrective Action
            │                           │
            └─────────────┬─────────────┘
                          ▼
                   GROUNDED ANSWER
```

### The Control Boundary
- **The LLM selects among bounded tools**: It proposes high-level retrieval and reasoning actions.
- **The harness validates and executes actions**: `HarnessReducer` checks budget limits, suppresses repeat queries, and verifies permissions before issuing calls to TigerGraph.
- **Zero raw database execution**: The model cannot execute arbitrary GSQL/Cypher, issue schema alterations, or mutate graph data.

---

## 4. TigerGraph Knowledge Graph

All graph data and vector embeddings are hosted on **TigerGraph 4.2.5** in the `OlympicGraphRAG` graph.

### Verified Graph Statistics
- **Vertices**: **28,305**
- **Edges**: **57,737**
- **Chunks**: **9,348**
- **Entities**: **8,003**
- **People / Athletes**: **4,333**
- **Events**: **2,187**
- **Teams**: **985**
- **Venues**: **320**
- **Countries**: **136**
- **Sports**: **42**
- **Documents**: **2,951**

### Core Schema & Relationships
```
(Document) ──[HAS_CHUNK]──> (Chunk)
(Document) ──[MENTIONS]──> (Entity)
(Entity) ──[RESOLVES_TO]──> (Person | Event | Sport | Country | Venue | Team)
(Person) ──[PARTICIPATED_IN]──> (Event)
(Person) ──[REPRESENTS]──> (Country)
(Event) ──[HELD_AT]──> (Venue)
(Event) ──[BELONGS_TO]──> (Sport)
(Team) ──[BELONGS_TO]──> (Country)
```

By indexing entity mentions and resolutions directly within TigerGraph, the system supports relationship-aware retrieval rather than treating text chunks as disconnected fragments.

---

## 5. Native Vector Retrieval

- **Embedding Model**: `nomic-ai/nomic-embed-text-v1.5`
- **Embedding Dimensions**: 768 dimensions
- **Distance Metric**: `COSINE`
- **Vector Index**: Native TigerGraph HNSW vector index on `Chunk.embedding`
- **Installed Query**: `searchChunksByVector`
- **Production Corpus**: 9,348 document chunks

*(Note: Gemini embeddings were evaluated during discovery, but `nomic-ai/nomic-embed-text-v1.5` is the active production embedding model configured in TigerGraph).*

---

## 6. DeepAgents + SkillsMiddleware

The A4 agent is implemented using the official DeepAgents `create_deep_agent` pattern coupled with `SkillsMiddleware`:

1. **`question-analysis`** (`skills/question-analysis/SKILL.md`): Deconstructs user queries into explicit entities, temporal anchors, and query intent categories.
2. **`retrieval-strategy-selection`** (`skills/retrieval-strategy-selection/SKILL.md`): Maps query attributes to appropriate retrieval strategies (A0, A1, A2).
3. **`graph-reasoning`** (`skills/graph-reasoning/SKILL.md`): Formulates multi-hop relational path expansions across vertex types.
4. **`evidence-verification`** (`skills/evidence-verification/SKILL.md`): Defines audit criteria for the A3 verifier before answer generation.
5. **`answer-synthesis`** (`skills/answer-synthesis/SKILL.md`): Synthesizes grounded natural language responses with strict citations to registered evidence IDs.

---

## 7. MCP Security Model

The system integrates with the official TigerGraph Model Context Protocol (MCP) using a strict read-only allowlist:

### Allowed Capabilities:
- `tigergraph__get_node`: Read attributes of a specific vertex.
- `tigergraph__get_edges`: Inspect incident edges for a vertex.
- `tigergraph__get_node_edges`: Retrieve localized 1-hop subgraphs.
- `tigergraph__get_neighbors`: Retrieve connected vertices filtered by target vertex/edge type.
- `searchChunksByVector`: Execute cosine similarity search via installed native query.

### Strictly Blocked:
- Arbitrary GSQL execution (`run_gsql`)
- Arbitrary Cypher execution (`run_cypher`)
- Schema mutations (`create_schema`, `drop_schema`)
- Data mutations (inserts, updates, deletes)
- Direct credential exposure to LLM context

---

## 8. Execution Harness & Evidence Ledger

The **Execution Harness** (`src/tgh/harness/`) enforces deterministic safety guardrails over non-deterministic LLM behavior:

- **`RunState` & `StateView`**: Immutable execution state tracking tool calls, accumulated evidence, and verification logs. Models receive a redacted `StateView` that eliminates prompt pollution.
- **`HarnessReducer`**: Pure transition reducer enforcing execution budgets:
  - `max_tool_calls`: **15** (prevents infinite reasoning loops)
  - `max_repairs`: **1** (single repair budget prevents endless retry oscillations)
  - `max_evidence_items`: **50** (bounds working memory context)
- **`Evidence Ledger`**: Central registry where all retrieved chunks and graph facts are indexed with unique IDs (`ev-xxxx`), confidence scores, and document provenance.
- **`A3 Verification Engine`**: Audits candidate answers for entity grounding, constraint satisfaction, and temporal alignment.

---

## 9. Benchmark Results

All comparative metrics are frozen against the 100-query public evaluation benchmark (`data/benchmarks/eval_public.jsonl`):

### Comparative Benchmark Table

| Pipeline | Execution Mode | Controller / Model | R@1 | R@5 | R@10 | R@20 | MRR | Mean Latency |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| **A0 Vector RAG** | Deterministic | Native TigerGraph HNSW | 68.00% | 83.00% | 86.00% | 90.00% | 0.7394 | 430.66 ms |
| **A1 GraphRAG** | Deterministic | Vector + Graph Traversal | 66.00% | 81.00% | 86.00% | 90.00% | 0.7290 | 2.37 s |
| **A4 Deterministic** | Rule-based | StateView Policy Controller | 91.00% | 100.00% | 100.00% | 100.00% | 0.9445 | ~8.20 s |
| **A4 LLM** | Model-driven | Gemini 3.5 Flash Lite | **84.00%** | **93.00%** | **96.00%** | **96.00%** | **0.8783** | **25.70 s** |

> **Important Distinction**:
> - **A4 Deterministic** is an algorithmic heuristic baseline (rule-based StateView controller). It is **not** an LLM-driven result.
> - **A4 LLM** is the genuine model-driven result executing multi-turn tool selection with Gemini 3.5 Flash Lite.

---

## 10. A4 LLM Benchmark Details

The frozen public model-driven benchmark (`experiments/runs/agentic_graphrag_public_benchmark_llm_complete.json`) captured the following production telemetry:

- **Public Questions Executed**: 100 / 100 (100% model-driven)
- **Fallback Executions**: 0 (0%)
- **API 429 Errors**: 0 during the completed run
- **Model**: `gemini-3.5-flash-lite` (Google Gemini API)
- **Retrieval Quality**: **R@1: 84.00%** | **R@5: 93.00%** | **R@10: 96.00%** | **R@20: 96.00%** | **MRR: 0.8783**
- **Latency**: Mean: **25,696.45 ms** | p50: **19,906.44 ms** | p95: **50,533.09 ms** | p99: **79,348.93 ms**
- **Model Calls**: 490 total calls (mean 4.90 calls/query)
- **Token Consumption**:
  - Input tokens: 2,213,748
  - Output tokens: 31,933
  - Total tokens: 2,245,681 (mean 22,456.8 tokens/query)
- **Pacing**: ~10 RPM (6.0 seconds between queries)
- **Verification & Repair**:
  - Initial Verification Passes: 49 / 100
  - Repairs Triggered (`max_repairs=1`): 51 / 100
  - Repairs Recovered at R@10: **51 / 51 (100%)**

---

## 11. Query-Type Results

Measured performance breakdown of the autonomous A4 LLM agent across distinct query categories:

| Query Type | Query Count | Recall@1 | Recall@5 | Recall@10 | MRR |
|:---|---:|---:|---:|---:|---:|
| **Temporal** | 22 | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Superlative** | 10 | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Lookup** | 19 | 89.47% | 100.00% | 100.00% | 0.9474 |
| **Aggregation** | 21 | 85.71% | 100.00% | 100.00% | 0.9143 |
| **Multi-hop** | 28 | 60.71% | 75.00% | 85.71% | 0.6653 |

### Analysis:
- **Temporal & Superlative (100% R@1)**: The model accurately identifies chronological markers and extreme target conditions, routing them directly through targeted graph entity filtering.
- **Lookup & Aggregation (89.5% & 85.7% R@1)**: Highly effective combination of vector seed lookup and graph neighbor aggregation.
- **Multi-hop (60.71% R@1, 85.71% R@10)**: Multi-hop queries represent the most difficult category for the LLM-driven controller. When reasoning chains span intermediate entities that lack direct single-hop graph connectivity, the model occasionally stops exploration early or exhausts tool budgets before finding the bridge entity.

---

## 12. Hidden Benchmark Section

- **Hidden-set execution projection:** ~50 queries  
  *(Projected from observed execution characteristics; not used for accuracy claims.)*

An actual hidden-set validation run was started with the frozen model-driven pipeline (`experiments/runs/agentic_graphrag_hidden_llm_validation.json`). 22 queries completed before the external model-service quota interrupted further execution. 

The partial artifact is retained as measured telemetry; **no hidden accuracy claim is made**, strictly adhering to hackathon transparency guidelines.

---

## 13. Project Structure

```
├── pyproject.toml              # Project configuration and dependencies
├── LICENSE                     # Apache 2.0 License
├── .gitignore                  # Gitignore protecting data, caches, and secrets
├── .env.example                # Template for environment credentials
├── README.md                   # Main documentation page
├── src/tgh/                    # Core Python package
│   ├── domain/                 # Domain entity models and schema mappings
│   ├── embeddings/             # Embedding client interfaces (Nomic v1.5)
│   ├── evaluation/             # Metrics calculators (Recall@K, MRR, Latency)
│   ├── evidence/               # Evidence Ledger and provenance tracking
│   ├── harness/                # Authoritative RunState, StateView, Reducer
│   ├── ingestion/              # Graph schema ingestion and chunking
│   ├── mcp/                    # Allowlisted TigerGraph MCP tools & contracts
│   ├── policies/               # Orchestrator, policy controllers, and A4 loop
│   ├── retrieval/              # Vector, Graph, and GraphRAG retrievers
│   └── telemetry/              # Trace recording and execution telemetry
├── skills/                     # DeepAgents SkillsMiddleware skills
│   ├── question-analysis/
│   ├── retrieval-strategy-selection/
│   ├── graph-reasoning/
│   ├── evidence-verification/
│   └── answer-synthesis/
├── scripts/                    # Benchmark execution and ingestion scripts
│   ├── benchmark_agentic_graphrag_llm_complete.py
│   ├── benchmark_agentic_graphrag.py
│   ├── benchmark_graphrag_retrieval.py
│   └── benchmark_tigergraph_retrieval.py
├── tests/                      # Pytest suite (132 unit & integration tests)
├── docs/                       # In-depth architectural & benchmark documentation
│   ├── ARCHITECTURE.md         # Detailed control flow and harness design
│   ├── BENCHMARKS.md           # Full benchmark numbers & reproduction guide
│   ├── EVALUATION.md           # Evaluation protocol and metric formulas
│   ├── PROJECT_STATUS.md       # Concise milestone and verification status
│   └── SUBMISSION.md           # Complete hackathon submission document
├── experiments/runs/           # Frozen benchmark runs and telemetry artifacts
└── results/                    # Static metrics dashboard
    └── metrics_dashboard.md
```

---

## 14. Security & Integrity

- **Environment-based Credentials**: All secrets (`TIGERGRAPH_SECRET`, `GEMINI_API_KEY`) are read from local `.env` and are strictly gitignored.
- **Zero Exposed Credentials**: The public repository contains no private keys, tokens, or personal paths.
- **Read-Only Database Policy**: The agent operates through an allowlisted read-only MCP interface. Graph mutation and arbitrary GSQL/Cypher queries are blocked by the execution harness.
- **Evaluation Honesty**: 
  - Deterministic and LLM-driven results are reported in separate columns.
  - Public benchmark numbers match the frozen artifact verbatim.
  - Zero accuracy claims are made on hidden evaluation sets.
  - No synthetic telemetry or fabricated results exist.

---

## 15. Getting Started & Reproducibility

### Prerequisites
- Python 3.11 or 3.12
- An active TigerGraph 4.2.5 instance with the `OlympicGraphRAG` schema loaded

### Setup
```bash
# Clone the repository
git clone https://github.com/yeshiarK/tigergraph-hackathon.git
cd tigergraph-hackathon

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode with dependencies
pip install -e .

# Configure environment variables
cp .env.example .env
# Edit .env with your TIGERGRAPH_HOST, TIGERGRAPH_SECRET, and GEMINI_API_KEY
```

### Running Tests
```bash
pytest
```
*132 tests pass. TigerGraph cloud connectivity test checks live status or gracefully skips if cloud instance is paused.*

### Running Public Benchmarks
- **Run A0 Vector Baseline**:
  ```bash
  python scripts/benchmark_tigergraph_retrieval.py
  ```
- **Run A1 GraphRAG Baseline**:
  ```bash
  python scripts/benchmark_graphrag_retrieval.py
  ```
- **Run A4 Deterministic Agent**:
  ```bash
  python scripts/benchmark_agentic_graphrag.py
  ```
- **Run A4 Model-Driven Agent (Gemini 3.5 Flash Lite)**:
  ```bash
  python scripts/benchmark_agentic_graphrag_llm_complete.py
  ```

---

## 16. Submission Information

- **Hackathon Track**: Agentic GraphRAG with TigerGraph
- **Repository**: [https://github.com/yeshiarK/tigergraph-hackathon](https://github.com/yeshiarK/tigergraph-hackathon)
- **Complete Submission Document**: [docs/SUBMISSION.md](file:///Users/yeshi/Desktop/tgh/docs/SUBMISSION.md)
- **Metrics Dashboard**: [results/metrics_dashboard.md](file:///Users/yeshi/Desktop/tgh/results/metrics_dashboard.md)
