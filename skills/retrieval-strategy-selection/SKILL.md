---
name: retrieval-strategy-selection
description: Dynamically selects the optimal retrieval policy among A0 Vector RAG, A1 Adaptive GraphRAG, and A2 Deterministic Graph Computation based on question structure.
allowed-tools:
  - tigergraph__search_top_k_similarity
  - tigergraph__run_installed_query
---

# Retrieval Strategy Selection Skill

## Purpose
This skill guides the agent in selecting the most accurate and cost-effective retrieval strategy:
- `A0_VECTOR_RAG`: Use for direct semantic queries, uncomplicated lookups, or pure temporal comparisons where dense chunk similarity provides high confidence.
- `A1_ADAPTIVE_GRAPHRAG`: Use when the question involves athlete participation or team representation where 1-hop or 2-hop entity context is beneficial, but strict structural constraints are absent.
- `A2_DETERMINISTIC_GRAPH`: Use when the question specifies structural graph constraints such as:
  - Venue-to-event multi-hop lookups ("held at <venue> on <date>/in <year>")
  - Country representation counts ("how many nations competed in...")
  - Multi-event participant disambiguation (e.g. Michael Phelps multiple events in same games)

## Decision Rules
1. **Rule 1 (Venue & Aggregation Priority)**:
   - If the query mentions a venue location or requests counting distinct nations/countries, route to `A2_DETERMINISTIC_GRAPH`.
2. **Rule 2 (Relational Entity Context)**:
   - If the query focuses on athlete representation ("represented <country>", "won medal in the event"), route to `A1_ADAPTIVE_GRAPHRAG`.
3. **Rule 3 (Default Semantic Baseline)**:
   - Otherwise, route to `A0_VECTOR_RAG` to preserve high accuracy and avoid unnecessary graph latency.
