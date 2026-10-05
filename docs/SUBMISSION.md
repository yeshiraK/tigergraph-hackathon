# TigerGraph Agentic GraphRAG — Hackathon Submission

**Project Title**: Agentic GraphRAG with TigerGraph  
**One-Line Pitch**: Adaptive multi-strategy GraphRAG combining native TigerGraph vector search, topological graph reasoning, DeepAgents orchestration, and bounded evidence verification.  
**Repository**: [https://github.com/yeshiarK/tigergraph-hackathon](https://github.com/yeshiarK/tigergraph-hackathon)

---

## 1. Problem Statement

Standard Retrieval-Augmented Generation (Vector RAG) assumes that every question can be resolved by retrieving top-$K$ text chunks via semantic embedding similarity. In complex relational domains like the Olympic Games, this assumption breaks down:
- **Relational Disconnect**: Vector embeddings capture topical similarity but cannot traverse multi-hop entity graphs (e.g. connecting a venue to an event, then to an athlete, and identifying their medal).
- **Structured Aggregations & Temporal Filters**: Vector math cannot execute deterministic count aggregations or enforce temporal filters (e.g. counting participating countries or filtering by Olympic year).
- **Hallucination Without Verification**: Unbounded LLM retrieval often presents plausible text without factual grounding, lacking an immutable ledger that links claims to verified evidence.

---

## 2. Solution: Agentic GraphRAG

Instead of applying a single static retrieval method, **Agentic GraphRAG** dynamically inspects runtime query requirements and chooses among specialized capabilities:
- **A0 (Semantic Vector RAG)**: Native TigerGraph HNSW vector search for broad topical inquiries.
- **A1 (Hybrid GraphRAG)**: Seed-based relational expansion linking vector chunks with connected entities.
- **A2 (Deterministic Graph Reasoning)**: Specialist graph algorithms for topological counting, multi-hop path traversal, and structured entity lookups.
- **A3 (Evidence Verification & Bounded Repair)**: Strict verification against an immutable Evidence Ledger, with at most one bounded corrective retrieval pass.
- **A4 (Agentic GraphRAG Orchestration)**: An autonomous controller powered by **DeepAgents** and **SkillsMiddleware** that dynamically selects tools, evaluates observations, and coordinates repair.

---

## 3. Why TigerGraph?

TigerGraph serves as the unified foundation for both graph topology and vector search:
- **High-Performance Native Graph Engine**: Houses 28,305 vertices and 57,737 edges across 9 entity types (Documents, Chunks, Entities, Events, People, Countries, Sports, Venues, Teams).
- **Native Vector Indexing**: High-performance native HNSW vector index executing 768-dimensional cosine distance similarity directly alongside graph topology via `searchChunksByVector`.
- **Hybrid Traversal**: Enables single-hop and multi-hop queries that transition seamlessly between unstructured text chunk embeddings and structured relational entity graphs.

---

## 4. Agentic Behavior & Control Boundary

A core architectural principle of this system is that **the LLM is an orchestrator, not an unbounded database administrator**:
- **DeepAgents + SkillsMiddleware**: The model reasons through five dedicated modular skills (`question-analysis`, `retrieval-strategy-selection`, `graph-reasoning`, `evidence-verification`, `answer-synthesis`).
- **Bounded Execution Harness**: All actions are validated and recorded by an authoritative `HarnessReducer` enforcing strict operational budgets (`max_tool_calls=15`, `max_repairs=1`, `max_evidence_items=50`).
- **Strict MCP Security Model**: The agent interfaces with TigerGraph exclusively through allowlisted Model Context Protocol (MCP) endpoints (`get_node`, `get_edges`, `get_node_edges`, `get_neighbors`, `searchChunksByVector`). Arbitrary GSQL, Cypher, and database mutations are strictly blocked.

---

## 5. System Architecture

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
                          │
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

---

## 6. Benchmark Results

Measured against the 100-query public evaluation benchmark (`data/benchmarks/eval_public.jsonl`):

### Comparative Performance Table

| Pipeline | Mode | Controller | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Mean Latency |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| **A0 Vector RAG** | Deterministic | Native TigerGraph HNSW | 68.00% | 83.00% | 86.00% | 90.00% | 0.7394 | 430.66 ms |
| **A1 GraphRAG** | Deterministic | Vector + Graph Traversal | 66.00% | 81.00% | 86.00% | 90.00% | 0.7290 | 2.37 s |
| **A4 Deterministic** | Rule-based | StateView Policy Controller | 91.00% | 100.00% | 100.00% | 100.00% | 0.9445 | ~8.20 s |
| **A4 LLM** | Model-driven | Gemini 3.5 Flash Lite | **84.00%** | **93.00%** | **96.00%** | **96.00%** | **0.8783** | **25.70 s** |

*Note: The deterministic A4 result is an algorithmic heuristic baseline and not an LLM-driven result. The A4 LLM result represents the genuine model-driven Gemini 3.5 Flash Lite implementation.*

### A4 LLM Query-Type Breakdown

| Query Type | Query Count | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
|:---|---:|---:|---:|---:|---:|---:|
| **Temporal** | 22 | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Superlative** | 10 | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **Lookup** | 19 | 89.47% | 100.00% | 100.00% | 100.00% | 0.9474 |
| **Aggregation** | 21 | 85.71% | 100.00% | 100.00% | 100.00% | 0.9143 |
| **Multi-hop** | 28 | 60.71% | 75.00% | 85.71% | 85.71% | 0.6653 |
| **Overall** | **100** | **84.00%** | **93.00%** | **96.00%** | **96.00%** | **0.8783** |

---

## 7. Hidden Evaluation Methodology

- **Hidden-set execution projection:** ~50 queries  
  *(Projected from observed execution characteristics; not used for accuracy claims.)*

An actual hidden-set validation run was started with the frozen model-driven pipeline. 22 queries completed before external model-service quota interrupted further execution. The partial artifact is retained as measured telemetry; **no hidden accuracy claim is made**, adhering to absolute evaluation integrity.

---

## 8. Technical Stack

- **Graph & Vector Database**: TigerGraph 4.2.5 (`OlympicGraphRAG`), Native HNSW Vector Index
- **Embedding Model**: `nomic-ai/nomic-embed-text-v1.5` (768-dimensional, COSINE metric)
- **Agent Framework**: DeepAgents (`create_deep_agent`) with `SkillsMiddleware`
- **Foundation LLM**: Google Gemini 3.5 Flash Lite (via Google Gemini API)
- **Protocol**: Model Context Protocol (MCP) with allowlisted read-only tools
- **Testing & Verification**: Pytest, Custom Execution Harness, Evidence Ledger

---

## 9. Key Innovations

1. **StateView & Execution Harness**: Decouples non-deterministic LLM planning from deterministic execution, enforcing bounded tool budgets and repeat-action suppression.
2. **Evidence Ledger with Bounded Repair**: Verifies claims against explicit registered chunks and facts, with a 100% repair recovery rate on public benchmark queries.
3. **Adaptive Multi-Strategy Routing**: Demonstrates that dynamic orchestration significantly outperforms both static vector search (+16% R@1) and static GraphRAG (+18% R@1).

---

## 10. Limitations

- **Multi-Hop Traversal Complexity**: Multi-hop queries requiring 3+ hops across sparsely connected subgraphs remain the primary bottleneck (60.71% R@1 vs 100% on temporal/superlative).
- **Service Latency**: Multi-turn model reasoning introduces additional wall-clock latency (mean 25.70s) compared to single-shot vector retrieval (430ms).

---

## 11. Reproducibility

Full setup instructions, test commands, and reproducible benchmark scripts are provided in [docs/BENCHMARKS.md](file:///Users/yeshi/Desktop/tgh/docs/BENCHMARKS.md) and [docs/EVALUATION.md](file:///Users/yeshi/Desktop/tgh/docs/EVALUATION.md).
