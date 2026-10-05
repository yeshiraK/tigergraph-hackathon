# Agentic GraphRAG with TigerGraph — Metrics Dashboard

This dashboard presents the frozen, verified experimental results and telemetry captured across retrieval architectures on the Olympic GraphRAG benchmark.

---

## 1. Headline Retrieval Metrics

Comparative evaluation across retrieval paradigms on the 100-question public evaluation benchmark (`data/benchmarks/eval_public.jsonl`):

| Pipeline | Mode | Controller / Model | R@1 | R@5 | R@10 | R@20 | MRR | Mean Latency |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|
| **A0 Semantic Vector RAG** | Deterministic | Native TigerGraph HNSW | 68.00% | 83.00% | 86.00% | 90.00% | 0.7394 | 430.66 ms |
| **A1 Hybrid GraphRAG** | Deterministic | Vector + Graph Traversal | 66.00% | 81.00% | 86.00% | 90.00% | 0.7290 | 2,370.00 ms |
| **A4 Agentic GraphRAG (Deterministic)** | Rule-based | StateView Policy Controller | 91.00% | 100.00% | 100.00% | 100.00% | 0.9445 | ~8,200.00 ms |
| **A4 Agentic GraphRAG (LLM)** | Model-Driven | Gemini 3.5 Flash Lite | **84.00%** | **93.00%** | **96.00%** | **96.00%** | **0.8783** | **25,696.45 ms** |

> **Methodological Note**: 
> The **A4 Deterministic** pipeline serves as an algorithmic upper-bound baseline using deterministic graph heuristics. 
> The **A4 LLM** pipeline is the genuine autonomous agent driven by Gemini 3.5 Flash Lite executing dynamic multi-turn tool selection via DeepAgents and SkillsMiddleware. These two results are kept strictly distinct.

---

## 2. A4 LLM Query-Type Performance Breakdown

Performance of the autonomous A4 LLM agent across distinct analytical query categories (100 public queries):

| Query Category | Query Count | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Latency (Mean) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Temporal** | 22 | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 25,616 ms |
| **Superlative** | 10 | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 23,284 ms |
| **Lookup** | 19 | 89.47% | 100.00% | 100.00% | 100.00% | 0.9474 | 22,231 ms |
| **Aggregation** | 21 | 85.71% | 100.00% | 100.00% | 100.00% | 0.9143 | 25,876 ms |
| **Multi-hop** | 28 | 60.71% | 75.00% | 85.71% | 85.71% | 0.6653 | 29,321 ms |
| **Total / Overall** | **100** | **84.00%** | **93.00%** | **96.00%** | **96.00%** | **0.8783** | **25,696.45 ms** |

### Key Observations:
1. **Perfect Precision on Temporal and Superlative Queries**: 100% R@1 on temporal and superlative queries demonstrate the model's ability to identify relevant years, editions, and extreme records by combining graph entity lookup with targeted chunk retrieval.
2. **High Reliability on Lookups and Aggregations**: 89.5% and 85.7% R@1 demonstrate effective routing between vector search and graph-neighbor aggregation.
3. **Multi-Hop Challenge**: Multi-hop queries represent the most complex reasoning path (60.71% R@1, 85.71% R@10). When a query spans entities not directly linked in 1-hop subgraphs, the agent must iteratively discover intermediate bridge entities.

---

## 3. Execution Latency Distribution (A4 LLM)

Measured latency distribution across all 100 public evaluation queries under full autonomous model-driven orchestration:

| Latency Metric | Measured Value | Notes |
|:---|---:|:---|
| **Mean Latency** | 25,696.45 ms (25.70 s) | Average end-to-end execution including model reasoning and TigerGraph round-trips |
| **Median (p50)** | 19,906.44 ms (19.91 s) | Typical 3-4 turn query |
| **p90** | 44,528.00 ms (44.53 s) | Queries requiring multi-step graph exploration |
| **p95** | 50,533.09 ms (50.53 s) | Queries with A3 verification failure and subsequent bounded repair |
| **p99** | 79,348.93 ms (79.35 s) | Complex multi-hop queries approaching max tool budget |
| **Min Latency** | 9,842.11 ms (9.84 s) | Fast single-turn lookup |
| **Max Latency** | 84,210.55 ms (84.21 s) | 5-step repair loop |

---

## 4. LLM Telemetry & Resource Consumption

Complete telemetry captured from the frozen production run (`experiments/runs/agentic_graphrag_public_benchmark_llm_complete.json`):

| Telemetry Parameter | Production Value | Per-Query Average |
|:---|---:|---:|
| **Total Public Queries** | 100 | - |
| **Model-Driven Executions** | 100 (100%) | - |
| **Deterministic Fallback Executions** | 0 (0%) | - |
| **API 429 / Rate Limit Errors** | 0 (0%) | - |
| **Underlying Foundation Model** | `gemini-3.5-flash-lite` | - |
| **Total Model Invocations** | 490 | 4.90 calls / query |
| **Total Prompt (Input) Tokens** | 2,213,748 | 22,137.5 tokens / query |
| **Total Completion (Output) Tokens** | 31,933 | 319.3 tokens / query |
| **Total Model Tokens** | 2,245,681 | 22,456.8 tokens / query |
| **Execution Pacing** | ~10 RPM (6.0s spacing) | Enforced to guarantee zero rate-limit drops |

---

## 5. Verification & Bounded Repair Performance (A3)

The A3 Verification Engine audits evidence grounding and structural integrity before emitting the final answer. When verification fails, the bounded repair engine triggers targeted retrieval:

| Verification Metric | Value | Rate |
|:---|---:|---:|
| **Queries Verified on First Pass** | 49 / 100 | 49.0% |
| **Repairs Triggered (`max_repairs=1`)** | 51 / 100 | 51.0% |
| **Repairs Successfully Recovered at R@10** | 51 / 51 | **100.0%** |
| **Budget Violations / Runaway Loops** | 0 / 100 | 0.0% |

All 51 queries triggering verification repair successfully incorporated corrective evidence into the Evidence Ledger and recovered relevant ground-truth context within the top-10 retrieved artifacts.

---

## 6. Hidden-Set Execution Validation (Telemetry Only)

| Parameter | Observed Status |
|:---|:---|
| **Hidden Evaluation Dataset** | `data/benchmarks/eval_hidden.jsonl` (unseen evaluation set) |
| **Execution Projection** | **~50 queries** (*projected from observed execution characteristics; not used for accuracy claims*) |
| **Executed Queries** | 22 queries completed model-driven before external model-service quota interrupted execution |
| **Execution Mode** | 100% model-driven (`gemini-3.5-flash-lite`), 0 fallbacks |
| **Measured Telemetry Artifact** | `experiments/runs/agentic_graphrag_hidden_llm_validation.json` |
| **Hidden Accuracy Claim** | **NOT CLAIMED** (in strict adherence to hackathon integrity guidelines) |

---

## 7. Engineering Constraints & Safety Limits

All agent executions are constrained by immutable safety parameters in `src/tgh/harness/reducer.py`:

- `max_tool_calls`: **15** (hard stop prevents infinite agent loops)
- `max_repairs`: **1** (single targeted repair iteration prevents oscillation)
- `max_evidence_items`: **50** (bounds working memory and context window consumption)
- `allowed_tools`: Strict allowlist (`vector_search`, `graph_search`, `graph_reasoning`, `verify_evidence`)
- `database_mutation`: **BLOCKED** (no mutation, write, or arbitrary GSQL privileges)
