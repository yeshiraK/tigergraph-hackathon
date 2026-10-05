# Evaluation Methodology & Protocol

This document defines the formal evaluation methodology, metric calculations, query categorization, and reproducibility guidelines for the TigerGraph Agentic GraphRAG system.

---

## 1. Evaluation Objectives

The evaluation framework is designed to test:
1. **Retrieval Completeness (Recall@K)**: Can the system retrieve the ground-truth document chunk within its top-K candidates?
2. **Retrieval Rank Quality (MRR)**: How high in the candidate list is the first relevant piece of evidence placed?
3. **Reasoning Agility**: Does an agentic controller dynamically adapt retrieval actions to the complexity of the question?
4. **Verification & Grounding**: Can an independent verifier catch ungrounded evidence and repair retrieval before answering?

---

## 2. Benchmark Dataset

### Public Evaluation Set (`data/benchmarks/eval_public.jsonl`)
- **Size**: 100 queries with verified ground-truth chunk IDs and gold entity references.
- **Corpus**: Olympic Games knowledge domain (9,348 text chunks, 28,305 vertices, 57,737 edges).
- **Taxonomy**:
  - **Temporal (22 queries)**: Questions with explicit or implicit time constraints (e.g., "Which city hosted the Olympic Games in 1992?").
  - **Superlative (10 queries)**: Questions requiring extreme value discovery (e.g., "Who won the most gold medals in a single Games?").
  - **Lookup (19 queries)**: Direct entity-property lookups (e.g., "What venue hosted the swimming events in 2000?").
  - **Aggregation (21 queries)**: Questions requiring neighborhood aggregation (e.g., "How many sports were contested by the United States team?").
  - **Multi-hop (28 queries)**: Complex relational chains traversing multiple entity types (e.g., "Find the athlete who represented country X, won gold in sport Y, and competed at venue Z").

### Hidden Evaluation Set (`data/benchmarks/eval_hidden.jsonl`)
- Maintained as an unseen validation set to verify execution robustness under novel phrasing.
- In accordance with hackathon evaluation integrity, hidden benchmark gold labels are not exposed, and **no hidden accuracy claims are made**.

---

## 3. Metric Formulations

### Recall@K ($R@K$)
Measures the proportion of queries for which at least one ground-truth document chunk appears in the top $K$ retrieved chunks:
$$\text{Recall}@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(\text{rank}(q) \le K\right)$$
where $\text{rank}(q)$ is the 1-based rank of the highest-ranked ground truth chunk for query $q$, and $\mathbb{I}$ is the indicator function. Evaluated at $K \in \{1, 5, 10, 20\}$.

### Mean Reciprocal Rank (MRR)
Measures the reciprocal rank of the first relevant chunk returned:
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}(q)}$$
If no relevant chunk is retrieved within the candidate window, $\frac{1}{\text{rank}(q)} = 0$.

### Latency
Total wall-clock duration from initial query submission to final verified response, measured in milliseconds ($ms$). Includes model reasoning, API calls, TigerGraph traversals, and verification cycles.

---

## 4. Evaluation Harness & Boundaries

The evaluation harness (`src/tgh/harness/`) enforces strict operational limits:
- **`max_tool_calls = 15`**: Ensures bounded execution and prevents runaway agent loops.
- **`max_repairs = 1`**: Restricts verification repair loops to a single corrective action.
- **`max_evidence_items = 50`**: Caps working memory to maintain precision in context window.
- **`allowed_tools`**: Read-only operations (`vector_search`, `graph_search`, `graph_reasoning`, `verify_evidence`).
- **Database Safety**: All TigerGraph mutation or arbitrary query endpoints are strictly blocked.

---

## 5. Reproducing Benchmark Runs

### 1. Environment Setup
```bash
git clone https://github.com/yeshiarK/tigergraph-hackathon.git
cd tigergraph-hackathon
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configure Environment
Create `.env` using `.env.example`:
```bash
cp .env.example .env
```
Populate `.env` with:
```env
TIGERGRAPH_HOST=https://your-instance.i.tgcloud.io
TIGERGRAPH_GRAPH_NAME=OlympicGraphRAG
TIGERGRAPH_SECRET=your_secret_token
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Running Retrieval Baselines
- Run A0 (Semantic Vector RAG):
  ```bash
  python scripts/benchmark_tigergraph_retrieval.py
  ```
- Run A1 (Hybrid GraphRAG):
  ```bash
  python scripts/benchmark_graphrag_retrieval.py
  ```
- Run A4 Deterministic Agent:
  ```bash
  python scripts/benchmark_agentic_graphrag.py
  ```

### 4. Running Model-Driven A4 Benchmark (Gemini 3.5 Flash Lite)
```bash
python scripts/benchmark_agentic_graphrag_llm_complete.py
```
This script runs the full 100-question public benchmark with strict pacing (~10 RPM) and saves output telemetry to `experiments/runs/agentic_graphrag_public_benchmark_llm_complete.json`.
