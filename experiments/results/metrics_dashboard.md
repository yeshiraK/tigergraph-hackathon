# Olympic GraphRAG Benchmark Metrics Dashboard

## Comparative Retrieval Performance

Comprehensive evaluation over the 100-question public benchmark (`data/benchmarks/eval_public.jsonl`) comparing three retrieval and reasoning paradigms on the same TigerGraph Olympic knowledge graph.

| Pipeline | R@1 | R@5 | R@10 | R@20 | MRR | Mean Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0 (Standard Vector RAG)** | 68% | 83% | 86% | 90% | 0.7394 | 430.66 ms |
| **A1 (Adaptive GraphRAG)** | 66% | 81% | 86% | 90% | 0.7290 | 2373 ms |
| **A4 execution baseline** | **91%** | **100%** | **100%** | **100%** | **0.9445** | 7686.46 ms |

---

## Performance by Question Type (A4 Execution Baseline)

| Question Type | Count | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **temporal** | 22 | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 |
| **multi_hop** | 28 | 96.43% | 100.00% | 100.00% | 100.00% | 0.9732 |
| **lookup** | 19 | 89.47% | 100.00% | 100.00% | 100.00% | 0.9474 |
| **aggregation** | 21 | 85.71% | 100.00% | 100.00% | 100.00% | 0.9143 |
| **superlative** | 10 | 70.00% | 100.00% | 100.00% | 100.00% | 0.8000 |

---

## Technical Interpretation

1. **Topological Superiority on Multi-Hop Queries**:
   - Standard semantic vector search (A0) suffers from embedding disconnect when answering composite queries requiring relational linking (e.g. connecting a specific venue name to an Olympic event, then retrieving the medalist).
   - A4's dynamic orchestration selects structured TigerGraph traversals (A2) to achieve **96.43% R@1** and **100% R@5/10/20** on multi-hop questions.

2. **Self-Correction via Verification & Repair**:
   - The Evidence Ledger coupled with A3 verification catches under-supported candidate answers and initiates a single bounded repair cycle.
   - Across 53 queries requiring evidence repair, 100% recovered relevant gold documents within the top 10 ranked positions, elevating overall R@1 from 68% (A0) to **91%** (A4) and achieving **100% R@5/10/20**.

3. **Latency-Accuracy Trade-off**:
   - A0 provides sub-second latency (430.66 ms) suitable for trivial lookups.
   - A4 trades bounded execution time (~7.6s average) for verifiable factual groundedness and perfect top-5 retrieval recall across the 100 benchmark queries.
