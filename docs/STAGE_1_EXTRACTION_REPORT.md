# Stage 1 Corpus-Grounded Graph Extraction Report

**Date**: 2026-10-04  
**Status**: COMPLETE & VERIFIED (Offline Validation)  
**Live TigerGraph State**: UNCHANGED (0 writes executed)  
**Benchmark Constraint**: `data/benchmarks/eval_hidden.jsonl` was NOT accessed.

---

## 1. What Changed

Two targeted, corpus-grounded corrections were implemented in the deterministic extraction pipeline:

1. **Person vs. Team Classification (`is_team_event`)**:
   - Location: `src/tgh/ingestion/graph_extractor.py`
   - **Solo/Individual Precedence**: If the combined title and event text explicitly contains `"solo"` or `"individual"`, the event is classified as `Person` (single athlete), taking strict precedence over generic `"synchronized"` or `teams:` infobox heuristics.
   - **Multi-Crew Sailing / Gymnastics Recognition**: Added deterministic detection for known multi-crew classes (`"49er"`, `"470"`, `"nacra 17"`, `"nacra"`, `"yngling"`, and `"group"` in rhythmic gymnastics), properly classifying them as `Team`.
   - **Conservative Fallback**: Preserved existing conservative heuristics for single-handed boats (`Laser`, `Finn`) when `t_num == c_num`.

2. **Chunk-Level MENTIONS Provenance**:
   - Location: `src/tgh/ingestion/graph_extractor.py`
   - Replaced naive `#c0000` document-level linking with exact chunk text scanning using production `SemanticChunker(target_tokens=768)`.
   - `Chunk -(MENTIONS)-> Entity` edges are emitted **only** when the entity surface text (or 3-letter NOC code with word boundaries) physically appears within that specific chunk's text.
   - Multiple chunk occurrences create multiple provenance edges to the **same single canonical Entity vertex** (no vertex duplication).
   - **Title-Only Sport Provenance**: For 412 documents where the sport is derived solely from `Document.title` and absent from all body text chunks, **no fabricated chunk MENTIONS edge is created**. The sport retains its document/event level graph linkage via `Event -(BELONGS_TO)-> Sport` and `Entity -(RESOLVES_TO)-> Sport`.

---

## 2. Why Each Change Was Required

1. **Classification Rationale**:
   - The Stage 1 audit identified 13 solo sailing and synchronized swimming events (e.g. Ben Ainslie in Finn, Robert Scheidt in Laser, Kristen Babb-Sprague in Solo Synchronized Swimming) that were incorrectly categorized as `Team` because of competing infobox fields or `"synchronized"` in the sport name.
   - Conversely, two-person boats like 49er and 470 require crew coordination and roster medalists as a team unit.
   - Grounding these rules strictly in event titles and infobox patterns eliminates classification leakage without resorting to external knowledge or brittle fuzzy matching.

2. **MENTIONS Provenance Rationale**:
   - Vector search returns specific chunks (e.g., `#c0002` or `#c0003`). If `MENTIONS` edges only pointed to `#c0000`, a graph traversal starting from retrieved chunks would fail to find connected entities.
   - Conversely, connecting an entity to chunks where it is not mentioned introduces noise and corrupts the sub-graph retrieval context.
   - Title-only sports must not have fabricated edges to avoid hallucinating text mentions that do not exist.

---

## 3. Before vs. After Statistics

| Metric | Pre-Audit Baseline | Revised Post-Correction | Delta / Impact |
|:---|:---:|:---:|:---|
| **Total Documents Processed** | 2,951 | 2,951 | 0 (Exact corpus) |
| **Olympic Event Documents** | 2,187 | 2,187 | 0 (100% parsed) |
| **Skipped Non-Olympic Documents** | 764 | 764 | 0 (Expected) |
| **Event Vertices** | 2,187 | 2,187 | 0 |
| **Sport Vertices** | 42 | 42 | 0 |
| **Venue Vertices** | 320 | 320 | 0 |
| **Country Vertices** | 136 | 136 | 0 |
| **Person Vertices** | 4,324 | 4,333 | +9 (Recovered solo medalists) |
| **Team Vertices** | 993 | 985 | -8 (Net corrected rosters) |
| **Entity Vertices** | 7,994 | 8,003 | +9 (Clean 1:1 resolve) |
| **TOTAL PROPOSED VERTICES** | 15,996 | **16,006** | +10 net unique vertices |
| **BELONGS_TO Edges** | 2,187 | 2,187 | 0 |
| **HELD_AT Edges** | 2,125 | 2,125 | 0 |
| **PARTICIPATED_IN Edges** | 6,842 | 6,842 | 0 |
| **REPRESENTS Edges** | 5,351 | 5,351 | 0 |
| **MENTIONS Edges** | 10,879 | **23,881** | **+13,002 (+119.5% grounded coverage)** |
| **RESOLVES_TO Edges** | 7,994 | 8,003 | +9 |
| **TOTAL PROPOSED EDGES** | 35,378 | **48,389** | **+13,011 edges** |

---

## 4. Person / Team Correction Results

- **Total Medalist Classifications**: 6,843
  - **Person Classifications**: 5,813 (84.95%)
  - **Team Classifications**: 1,030 (15.05%)
- **Targeted Audit Cases Verified**:
  - `Synchronized swimming – Women's solo` -> **Person** (Fixed)
  - `Gymnastics – Women's artistic individual all-around` -> **Person** (Fixed)
  - `Sailing – Men's Finn` / `Laser` -> **Person** (Preserved)
  - `Sailing – 49er` / `470` / `Nacra 17` / `Yngling` -> **Team** (Fixed)
  - `Rhythmic gymnastics – Group all-around` -> **Team** (Fixed)
- **Referential Integrity**: 0 ambiguous or conflicting Person/Team IDs.

---

## 5. MENTIONS Provenance Results

- **Total MENTIONS Edges**: 23,881
- **Multi-Chunk Grounding**:
  - Entities appearing in 1 chunk: 5,062
  - Entities appearing in 2 chunks: 1,560
  - Entities appearing in 3 chunks: 500
  - Entities appearing in 4 chunks: 288
  - Entities appearing in $\ge$ 5 chunks: 593
- **Title-Derived Sports Absent from Chunk Text**: Exactly **412 documents** (18.84%).
  - Zero phantom chunk `MENTIONS` edges fabricated for these 412 cases.
  - Full document-level linkage preserved via `Event -(BELONGS_TO)-> Sport` and `Entity -(RESOLVES_TO)-> Sport`.

---

## 6. Remaining Unresolved Cases

All unresolved cases originate directly from raw Wikipedia infobox omissions and are handled safely without external guessing:

1. **Unresolved Venues (62 events)**: Missing `venue:` key in infobox (e.g. historical early 20th-century events).
2. **Unresolved Countries (30 medal slots)**: Incomplete or missing `NOC` code in older/disputed historical records.
3. **Unresolved Medalists (1 medal slot)**: Document `Q26212147` where the gold medal was stripped/vacated (`gold: "vacant"`).
4. **Ambiguous Mappings**: Exactly **0**.

---

## 7. Confirmation: Live TigerGraph Unchanged

A live read of the TigerGraph Cloud instance (`OlympicGraphRAG`) confirmed:
- `Document`: 2,951 (Unchanged)
- `Chunk`: 9,348 (Unchanged)
- `HAS_CHUNK`: 9,348 (Unchanged)
- `Event`: 0 (Unchanged)
- `Sport`: 0 (Unchanged)
- `Venue`: 0 (Unchanged)
- `Country`: 0 (Unchanged)
- `Person`: 0 (Unchanged)
- `Team`: 0 (Unchanged)
- `Entity`: 0 (Unchanged)
- All Stage 1 edges: 0 (Unchanged)

**No TigerGraph write operations were executed.**

---

## 8. Confirmation: Hidden Benchmark Isolation

- `data/benchmarks/eval_hidden.jsonl` was **NOT read, parsed, loaded, or accessed** at any point during extraction, validation, or testing.
- All testing and validation relied exclusively on `data/raw/corpus.jsonl` and synthetic unit test fixtures.
