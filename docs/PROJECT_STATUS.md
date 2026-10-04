# Project Status Record

## Current Phase
**Stage 3 / A2: Deterministic Graph-Computation Tools Complete & Verified**

---

## Verified Environment Details

- **Project Root**: `/Users/yeshi/Desktop/tgh`
- **Python Runtime**: `Python 3.12.0` (`.venv` active)
- **Dependencies**: `tgh 0.1.0`, `pyTigerGraph 1.6.0`, `nomic 3.6.0`, `torch 2.6.0`, `pytest 9.1.1`, `ruff 0.16.10`
- **Target Graph**: `OlympicGraphRAG` on TigerGraph Cloud
- **Vector Index**: 768D HNSW COSINE on `Chunk.embedding` (Nomic v1.5)
- **Live Database Contents**:
  - `Document`: 2,951
  - `Chunk`: 9,348
  - `HAS_CHUNK`: 9,348
  - `Event`: 2,187
  - `Person`: 4,333
  - `Team`: 985
  - `Sport`: 42
  - `Venue`: 320
  - `Country`: 136
  - `Entity`: 8,003
  - `BELONGS_TO`: 2,187
  - `HELD_AT`: 2,125
  - `PARTICIPATED_IN`: 6,842
  - `REPRESENTS`: 5,351
  - `MENTIONS`: 23,881
  - `RESOLVES_TO`: 8,003
  - Total Vertices: 28,305 | Total Edges: 57,737

---

## Completed Milestones

### 1. Stage 1 Live Ingestion (Complete & Frozen)
- Ingested 16,006 vertices and 48,389 edges into `OlympicGraphRAG` without modifying the live schema.
- Preserved existing corpus layer: 2,951 `Document` and 9,348 `Chunk` vertices with 768D Nomic vectors and HNSW index.
- 100% referential integrity verified across all 6 domain edge types.
- Grounded `MENTIONS` provenance verified (4,755 mentions beyond chunk 0; 0 synthetic phantom mentions for title-only sports).

### 2. Stage 2 GraphRAG Retrieval Implementation (Complete & Frozen)
- Implemented `src/tgh/retrieval/`:
  - `models.py`: Typed models (`SeedChunk`, `DiscoveredEntity`, `DomainVertex`, `GraphPath`, `EvidenceChunk`, `GraphRAGRetrievalResult`).
  - `vector.py`: `TigerGraphVectorRetriever` interfacing `searchChunksByVector`.
  - `graph.py`: `TigerGraphTraverser` performing bounded multi-hop expansion with URL-safe entity queries, connection pooling, and in-memory LRU edge caching.
  - `graphrag.py`: `GraphRAGRetriever` deterministic fusion, adaptive gating, and ranking coordinator.
- Adaptive GraphRAG Public Benchmark (100 Questions):
  - Recall@1 = 0.6600, Recall@5 = 0.8100, Recall@10 = 0.8600, Recall@20 = 0.9000, MRR = 0.7290.
  - Multi-Hop Recall@10 = 0.5357, Multi-Hop MRR = 0.2510.
  - Saved to `experiments/runs/tigergraph_adaptive_graphrag_public_benchmark.json`.

### 3. Stage 3 / A2 Deterministic Graph-Computation Tools (Complete & Verified)
- Implemented `src/tgh/telemetry/`:
  - `trace.py`: `OperationTrace` and `TraceRecorder` internal execution telemetry tracking `run_id`, operation name, inputs, latency_ms, success/failure, provenance, and structured error logs.
- Implemented `src/tgh/mcp/`:
  - `contracts.py`: Typed domain representations (`EventCandidate`, `EventContext`, `CountryRepresentation`, `CountryAggregation`, `ToolExecutionResult`).
  - `graph_tools.py`: `TigerGraphTools` implementing 6 exact graph primitives + 1 composite multi-hop reasoning capability:
    1. `get_events_for_participant(participant_type, participant_id)`: Person/Team -> PARTICIPATED_IN -> Event.
    2. `get_event_context(event_id)`: Event -> Sport, Venue, and reverse PARTICIPATED_IN participants.
    3. `get_country_for_participant(participant_type, participant_id)`: Person/Team -> REPRESENTS -> Country.
    4. `get_events_for_venue(venue_id, year, sport, event_name_query)`: Venue <- reverse_HELD_AT <- Event with early filtering.
    5. `filter_events(event_ids, year, sport, venue_id, event_name_query)`: Exact deterministic filtering on graph relationships.
    6. `aggregate_countries_for_participants(participants)`: Person/Team -> REPRESENTS -> Country deduplication.
    7. `execute_composite_multihop(...)`: Constraint intersection across participant, venue, year, sport, event-name, participant retrieval, and country aggregation.
- Unit Test Suite: 17/17 tests passing in `tests/test_graph_tools.py` (total 80 passing tests project-wide).
- Live Smoke Test: 6/6 public multi-hop benchmark questions successfully reached gold events in `experiments/runs/a2_graph_tools_smoke.json`.

---

## Live A2 Smoke Test Results (Targeted Public Questions)

| Question ID | Question Summary | Input Constraints | Gold Target | Gold Reached? | Vector Alone Rank | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pub-014** | Venue Riocentro Pavilion 4 -> 2016 Badminton | Venue: `venue_riocentro_pavilion_4`, Year: 2016, Sport: `badminton`, Name: `women's singles` | `Q25301483` (Carolina Marín) | **REACHED** (1 event isolated) | Rank 17 (Missed @ 10) | 6.68 s |
| **pub-028** | Olympic Aquatic Centre 2004 -> Disambiguate Phelps | Participant: `person_michael_phelps`, Year: 2004, Name: `400 metre individual medley` | `Q1141105` (Michael Phelps) | **REACHED** (1 event isolated) | Seed diluted (6 co-events) | 7.53 s |
| **pub-015** | London Velopark 2012 -> Cycling Pursuit | Venue: `venue_london_velopark`, Year: 2012, Sport: `cycling`, Name: `team pursuit` | `Q2297633` (King, Trott, Rowsell) | **REACHED** (2 events isolated) | Unranked (Missed @ 20) | 9.56 s |
| **pub-011** | Richmond Olympic Oval 2010 -> Speed Skating | Venue: `venue_richmond_olympic_oval`, Year: 2010, Sport: `speed skating`, Name: `3000 metres` | `Q580481` (Martina Sáblíková) | **REACHED** (1 event isolated) | Unranked (Missed @ 20) | 6.59 s |
| **pub-017** | Royal Artillery Barracks 2012 -> Shooting | Venue: `venue_royal_artillery_barracks`, Year: 2012, Sport: `shooting`, Name: `10 metre air rifle` | `Q1137721` (Yi Siling) | **REACHED** (2 events isolated) | Unranked (Missed @ 20) | 9.61 s |
| **pub-005** | Weightlifting Gymnasium 1988 -> 60 kg | Venue: `venue_olympic_weightlifting_gymnasium`, Year: 1988, Name: `60 kg` | `Q25239316` (Naim Süleymanoğlu) | **REACHED** (1 event isolated) | Rank 4 | 4.95 s |

**Summary**: 6 out of 6 targeted multi-hop questions (100%) successfully isolated and recovered the gold event vertex and participants, including 3 questions (`pub-011`, `pub-015`, `pub-017`) completely unranked by vector retrieval alone.

---

## Known Remaining Limitations

1. **Entity Name / Venue Extraction Requirement**:
   - The deterministic graph tools require an extracted entity identifier (e.g. `venue_riocentro_pavilion_4` or `person_michael_phelps`). When queries use colloquial aliases not resolved to an Entity hub, entity resolution upstream in the agent layer is required.
2. **Date String Granularity**:
   - While the graph holds `year` on Event vertices, exact calendar dates (e.g. "14 February" or "20 September") reside inside the document/chunk text rather than vertex integer attributes, so event name queries (e.g. "3000 metres") or chunk text verification are needed to narrow down events within the same venue and year.

---

### 4. Stage 4 / A3 Verification & Bounded One-Repair + A4 Agentic Orchestration (Complete & Verified)
- **Official TigerGraph MCP Integration**:
  - Layer 3 Client: `src/tgh/mcp/tigergraph_client.py` wrapping the official `tigergraph-mcp` (v1.0.3) package.
  - Exposes 10 strictly read/query/vector tools: `get_graph_schema`, `show_graph_details`, `get_node`, `get_nodes`, `get_node_edges`, `get_edges`, `get_neighbors`, `run_installed_query`, `search_top_k_similarity`, `fetch_vector`.
  - Blocks 24 unsafe mutation/DDL/raw-GSQL tools: `tigergraph__gsql`, `tigergraph__generate_gsql`, `tigergraph__create_graph`, `tigergraph__drop_graph`, `tigergraph__clear_graph_data`, `tigergraph__add_node(s)`, `tigergraph__delete_node(s)`, `tigergraph__add_edge(s)`, `tigergraph__delete_edge(s)`, etc.
- **Evidence Ledger (`src/tgh/evidence/ledger.py`)**:
  - Tracks discrete evidence items (`vector_chunk`, `graph_fact`, `graph_path`) with unique IDs, document IDs, provenance paths, and confidence scores. Authoritative support for all downstream answers.
- **Execution Harness (`src/tgh/harness/engine.py`)**:
  - Authoritative `RunState`, read-only `StateView`, and single-authority `HarnessReducer`. State can only be mutated through typed events (`STRATEGY_SELECTED`, `TOOL_CALLED`, `EVIDENCE_ADDED`, `REPAIR_ATTEMPTED`, `DOCS_RANKED`, `ANSWER_PROPOSED`, `EXECUTION_COMPLETED`).
- **A3 Verification & Bounded One-Repair (`src/tgh/policies/verification.py`)**:
  - Evidence-grounded verifier checking candidate answers, entities, and year constraints against ledger facts. Returns structured `SUPPORTED`, `INSUFFICIENT`, `CONTRADICTED`, or `REPAIR_REQUIRED`.
  - Enforces strict max 1 targeted repair attempt via `BoundedRepairExecutor` before finalizing.
- **A4 Agentic Orchestration (`src/tgh/policies/agentic_orchestrator.py`)**:
  - Dynamically routes queries between A0 Vector RAG, A1 Adaptive GraphRAG, and A2 Deterministic Graph Computation.
  - DeepAgents compatible workflow: Agent decides -> Tools execute -> TigerGraph provides facts -> Harness owns authoritative state -> Reducer applies transitions -> Ledger controls support.
- **Unit Test Suite**: 24/24 tests passing in `tests/test_stage4_agentic.py` (total 104 passing tests across the entire repository).
- **Live Smoke Test (6 Public Questions)**: Saved in `experiments/runs/a3_a4_smoke.json` (R@1 = 50%, R@5 = 83.3%, R@10 = 100.0%, 2 successful repairs).
- **Full Public Benchmark (100 Questions)**: Saved in `experiments/runs/agentic_graphrag_public_benchmark.json`.

---

## Public Benchmark Comparison (Full 100 Questions)

| Pipeline | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Multi-Hop R@10 | Multi-Hop MRR | Average Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0 Vector RAG Baseline** | 68.00% | 83.00% | 86.00% | 90.00% | 0.7394 | 50.00% | 0.2410 | 430.66 ms |
| **A1 Adaptive GraphRAG** | 66.00% | 81.00% | 86.00% | 90.00% | 0.7290 | 53.57% | 0.2510 | 2,370.00 ms |
| **A4 Agentic GraphRAG** | **69.00%** | **87.00%** | **94.00%** | **97.00%** | **0.7660** | **78.57%** | **0.3359** | 8,051.60 ms |

### Performance by Question Type (A4 Agentic GraphRAG)

| Question Type | Count | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **aggregation** | 21 | 85.71% | 100.00% | 100.00% | 100.00% | 0.9143 |
| **lookup** | 19 | 89.47% | 100.00% | 100.00% | 100.00% | 0.9474 |
| **temporal** | 22 | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **superlative** | 10 | 70.00% | 100.00% | 100.00% | 100.00% | 0.8000 |
| **multi_hop** | 28 | 17.86% | 53.57% | 78.57% | 89.29% | 0.3359 |

---

## Accuracy-First Diagnostic Insights

1. **A0 vs A4 Differences**:
   - Questions where A0 succeeded and A4 failed (@10): Only **1 question** (`pub-073`).
   - Questions where A4 succeeded and A0 failed (@10): **9 questions** (`pub-011`, `pub-014`, `pub-015`, `pub-017`, `pub-022`, `pub-050`, `pub-076`, `pub-081`, `pub-096`).
2. **A2 Graph Recovery**:
   - A2 deterministic graph computation successfully recovered **8 questions** that both A0 Vector RAG and A1 Adaptive GraphRAG missed entirely (`pub-011`, `pub-014`, `pub-015`, `pub-017`, `pub-022`, `pub-076`, `pub-081`, `pub-096`).
3. **Verification & One-Repair Impact**:
   - Verification flagged 53 questions as `REPAIR_REQUIRED`, triggering targeted one-repair fallback.
   - All 53 repaired queries successfully retained or recovered gold documents in the top 10 (100% recovery rate).
4. **Remaining Failures**:
   - Only 6 questions failed Recall@10 across the entire 100-question benchmark (`pub-023`, `pub-030`, `pub-038`, `pub-073`, `pub-079`, `pub-098`), all belonging to the complex multi-hop category with long venue/date phrasing.
5. **Latency Profile**:
   - Median (p50) latency is 3.66 s. The top latency contributors are broad venue traversals without year constraints (`pub-038`: 88.9 s, `pub-060`: 72.4 s).

---

## Known Remaining Limitations

1. **Colloquial Entity Name Aliases**:
   - When a question mentions a venue with non-standard abbreviations, exact vertex ID mapping requires entity resolution.
2. **Sub-Year Date String Filtering**:
   - Specific day/month tokens (e.g. "August 14") reside in chunk text; graph vertex attributes filter at integer year resolution.

---

## Verification Audit

- **Hidden Benchmark**: `eval_hidden.jsonl` was NOT accessed.
- **TigerGraph Cloud Schema**: Preserved intact (0 mutations, 0 deletions).
- **Git Commits**: Zero commits made (all changes in working tree).



