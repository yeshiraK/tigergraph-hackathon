# System Architecture & Technical Design

This document details the architectural design of the TigerGraph Agentic GraphRAG system, focusing on the DeepAgents orchestration layer, SkillsMiddleware, the deterministic Execution Harness, Evidence Ledger, and the TigerGraph MCP security boundary.

---

## 1. System Overview

Agentic GraphRAG combines the semantic search capabilities of dense vector retrieval with the relational power of graph databases and the dynamic reasoning of autonomous LLM agents. Rather than executing a rigid, one-size-fits-all retrieval pipeline, the system selects, executes, verifies, and repairs retrieval strategies based on runtime observations.

```
USER QUESTION
      │
      ▼
┌────────────────────────────────────────────────────────┐
│                      A4 AGENT                          │
│          DeepAgents + SkillsMiddleware                 │
└─────────────────────────┬──────────────────────────────┘
                          │ dynamic tool invocation
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
      [SUPPORTED]                   [REPAIR]
            │                           │ (bounded: max 1)
            │                           ▼
            │               Targeted Corrective Action
            │               (Refined search / expansion)
            │                           │
            └─────────────┬─────────────┘
                          ▼
                   GROUNDED ANSWER
```

---

## 2. DeepAgents & SkillsMiddleware

The autonomous agent is implemented using the official DeepAgents SDK (`create_deep_agent`) coupled with custom `SkillsMiddleware`. 

### The Five Core Skills

1. **`question-analysis`** (`skills/question-analysis/SKILL.md`):
   - Decomposes the user query into entity mentions, relationships, temporal constraints, and target query type (lookup, aggregation, multi-hop, superlative, temporal).
2. **`retrieval-strategy-selection`** (`skills/retrieval-strategy-selection/SKILL.md`):
   - Maps analyzed intent to initial retrieval actions (e.g. dense vector search for broad topical queries, entity neighbor exploration for structured lookups).
3. **`graph-reasoning`** (`skills/graph-reasoning/SKILL.md`):
   - Guides multi-hop path traversal, relational expansion, and entity disambiguation across TigerGraph vertices and edges.
4. **`evidence-verification`** (`skills/evidence-verification/SKILL.md`):
   - Instructs the verifier on audit criteria: ensuring every factual statement in candidate answers links directly to verified chunk or graph evidence.
5. **`answer-synthesis`** (`skills/answer-synthesis/SKILL.md`):
   - Structures the final user response, synthesizing graph facts and text excerpts with explicit citation of evidence IDs.

---

## 3. The Execution Harness

A critical contribution of this project is the **bounded execution harness** (`src/tgh/harness/`), which decouples the LLM from the raw database and guarantees safe, predictable agent behavior:

### Components:
- **`RunState`**: Immutable state representation tracking query metadata, tool invocation counts, accumulated evidence, verification attempts, and repair budgets.
- **`StateView`**: Read-only projection passed to the agent, containing summarized observations without polluting the prompt context with raw database dumps.
- **`HarnessReducer`**: Pure transition function that updates `RunState` upon each action, verifying budgets and enforcing termination conditions.
- **`Budget Enforcement`**:
  - `max_tool_calls`: **15** (hard stop preventing infinite loops)
  - `max_repairs`: **1** (single repair budget prevents endless retry cycles)
  - `max_evidence_items`: **50** (bounds working memory context)
- **`Repeat / No-Progress Protection`**: Prevents the agent from re-issuing duplicate queries or identical parameter sets.

---

## 4. Evidence Ledger & A3 Verification

The **Evidence Ledger** (`src/tgh/evidence/ledger.py`) serves as the single source of truth during an execution session.

### Ledger Mechanics:
- Every tool action (vector retrieval, graph neighborhood search, path query) registers its output into the ledger as structured items.
- Duplicate chunks and redundant facts are deduplicated by unique ID (`chunk_id`, vertex ID).
- Both dense text chunks and discrete relational triples (e.g., `(Athlete, PARTICIPATED_IN, Event)`) are indexed with provenance.

### A3 Verification Engine:
Before a response is finalized, the verifier validates the candidate answer:
1. **Grounding Check**: Verifies that claims are supported by ledger items.
2. **Entity Consistency**: Checks that resolved entities in the answer match the queried entities.
3. **Temporal Alignment**: Validates that dates, Olympic years, and edition numbers align with graph facts.
4. **Repair Loop**: If verification fails and the repair budget ($1$) has not been exhausted, a targeted corrective query is executed to resolve the missing evidence.

---

## 5. TigerGraph MCP Security Boundary

The agent communicates with TigerGraph through a secure Model Context Protocol (MCP) interface with a strict allowlist.

### Allowed Capabilities:
- `get_node`: Inspect node attributes for a specific vertex.
- `get_edges`: Inspect outgoing and incoming relationships.
- `get_node_edges`: Retrieve localized 1-hop subgraphs.
- `get_neighbors`: Retrieve connected vertices by edge type.
- `searchChunksByVector`: Execute cosine similarity search via TigerGraph's installed native HNSW query.

### Security Guarantees:
- **No Arbitrary Queries**: Arbitrary GSQL and Cypher queries are blocked.
- **No Database Mutation**: All write, update, delete, and schema-altering operations are forbidden.
- **Credential Isolation**: TigerGraph host credentials and auth tokens are encapsulated in the harness backend and never passed into LLM prompt contexts.
