"""Phase 6: Public benchmark evaluation using live TigerGraph HNSW vector retrieval.

Evaluates 100 public benchmark questions against 9,348 corpus chunks in TigerGraph.
Does NOT access eval_hidden.jsonl.
"""

import json
import time
from pathlib import Path

import numpy as np
import pyTigerGraph as tg

from tgh.embeddings.metrics import (
    compute_first_gold_rank,
    compute_mrr,
    compute_recalls,
    is_hit_at_k,
)
from tgh.embeddings.nomic import NomicEmbeddingProvider


def load_env() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    res = {}
    if env_file.is_file():
        with env_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    res[k.strip()] = v.strip()
    return res


def main():
    repo_root = Path(__file__).resolve().parent.parent
    public_file = repo_root / "data" / "benchmarks" / "eval_public.jsonl"

    print("=" * 70)
    print("PHASE 6: PUBLIC RETRIEVAL BENCHMARK ON TIGERGRAPH (100 QUESTIONS)")
    print("=" * 70)

    # 1. Connect to TigerGraph
    env = load_env()
    conn = tg.TigerGraphConnection(
        host=env["TIGERGRAPH_HOST"],
        graphname=env["TIGERGRAPH_GRAPH_NAME"],
        gsqlSecret=env["TIGERGRAPH_SECRET"],
    )
    print(f"Connected to graph: {conn.graphname}")

    # 2. Initialize Nomic Embedding Provider
    t_m0 = time.perf_counter()
    provider = NomicEmbeddingProvider(dimension=768)
    print(f"Loaded {provider.model_name} in {time.perf_counter() - t_m0:.2f}s")

    # 3. Load 100 public questions
    with public_file.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} public evaluation questions.")

    retrieved_docs_list = []
    gold_docs_list = []
    latencies_ms = []
    query_results = []

    print("\n--- Running 100-Query Retrieval Benchmark via TigerGraph HNSW ---")
    t_bench0 = time.perf_counter()

    for idx, q in enumerate(questions, start=1):
        qid = q["qid"]
        qtype = q["qtype"]
        question = q["question"]
        gold_set = set(q["gold_doc_ids"])
        gold_docs_list.append(gold_set)

        # Time the full retrieval request (Embedding + TigerGraph vector query)
        t_q0 = time.perf_counter()
        q_formatted = provider.format_query(question)
        q_vec = provider.embed_text(q_formatted)

        tg_res = conn.runInstalledQuery(
            "searchChunksByVector",
            params={"qvec": q_vec, "top_k": 20},
        )[0]
        q_latency = (time.perf_counter() - t_q0) * 1000.0
        latencies_ms.append(q_latency)

        distances = tg_res.get("@@distances", {})
        chunk_to_doc = tg_res.get("@@chunk_to_doc", {})
        candidates = tg_res.get("candidates", [])

        # Sort candidate chunks by ascending distance (descending cosine similarity)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: distances.get(c["v_id"], 1.0),
        )

        # Deduplicate to document level in order of best chunk rank
        seen_docs = set()
        ranked_docs = []
        for c in sorted_candidates:
            cid = c["v_id"]
            doc_id = chunk_to_doc.get(cid, "")
            if doc_id and doc_id not in seen_docs:
                seen_docs.add(doc_id)
                ranked_docs.append(doc_id)

        retrieved_docs_list.append(ranked_docs)

        first_rank = compute_first_gold_rank(ranked_docs, gold_set)
        hit_1 = is_hit_at_k(ranked_docs, gold_set, 1)
        hit_5 = is_hit_at_k(ranked_docs, gold_set, 5)
        hit_10 = is_hit_at_k(ranked_docs, gold_set, 10)

        query_results.append({
            "qid": qid,
            "qtype": qtype,
            "rank": first_rank,
            "hit_1": hit_1,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "latency_ms": q_latency,
        })

        if idx % 20 == 0 or idx == len(questions):
            curr_hits = sum(1 for r in query_results if r["hit_10"])
            print(
                f"  Processed {idx:3d}/100 questions | "
                f"Hit@10 so far: {curr_hits}/{idx} ({curr_hits/idx*100:.1f}%) | "
                f"Latency: {q_latency:.1f}ms"
            )

    total_bench_duration = time.perf_counter() - t_bench0

    # 4. Compute Benchmark Metrics
    recalls = compute_recalls(retrieved_docs_list, gold_docs_list, k_list=(1, 5, 10))
    mrr = compute_mrr(retrieved_docs_list, gold_docs_list)

    avg_latency = float(np.mean(latencies_ms))
    p50_latency = float(np.percentile(latencies_ms, 50))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))

    total_hit_any = sum(1 for r in query_results if r["rank"] is not None)
    total_hit_10 = sum(1 for r in query_results if r["hit_10"])

    print("\n" + "=" * 70)
    print("FINAL PUBLIC BENCHMARK RESULTS (TIGERGRAPH NATIVE HNSW RETRIEVAL)")
    print("=" * 70)
    print(f"Total Questions Evaluated:    {len(questions)}")
    print(
        f"Recall@1:                     {recalls.get(1, 0.0):.4f} "
        f"({recalls.get(1, 0.0)*100:.2f}%)"
    )
    print(
        f"Recall@5:                     {recalls.get(5, 0.0):.4f} "
        f"({recalls.get(5, 0.0)*100:.2f}%)"
    )
    print(
        f"Recall@10:                    {recalls.get(10, 0.0):.4f} "
        f"({recalls.get(10, 0.0)*100:.2f}%)"
    )
    print(f"Mean Reciprocal Rank (MRR):   {mrr:.4f}")
    print(f"Queries with Gold Hit (@10):  {total_hit_10} / {len(questions)}")
    print(f"Queries with Gold Hit (Top20): {total_hit_any} / {len(questions)}")
    print("\nLatency Statistics (End-to-End per Query):")
    print(f"  Average Latency:            {avg_latency:.2f} ms")
    print(f"  p50 (Median) Latency:       {p50_latency:.2f} ms")
    print(f"  p95 Latency:                {p95_latency:.2f} ms")
    print(f"  p99 Latency:                {p99_latency:.2f} ms")
    print(f"  Total Benchmark Duration:   {total_bench_duration:.2f} s")

    # 5. Breakdown by Question Type
    print("\n--- Performance by Question Type ---")
    types = sorted({q["qtype"] for q in questions})
    header = (
        f"{'Question Type':<20} | {'Count':<6} | {'Recall@1':<10} | "
        f"{'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<8}"
    )
    print(header)
    print("-" * 75)
    for qt in types:
        sub_indices = [i for i, q in enumerate(questions) if q["qtype"] == qt]
        sub_ret = [retrieved_docs_list[i] for i in sub_indices]
        sub_gold = [gold_docs_list[i] for i in sub_indices]
        sub_rec = compute_recalls(sub_ret, sub_gold, k_list=(1, 5, 10))
        sub_mrr = compute_mrr(sub_ret, sub_gold)
        line_str = (
            f"{qt:<20} | {len(sub_indices):<6} | {sub_rec.get(1, 0.0):<10.4f} | "
            f"{sub_rec.get(5, 0.0):<10.4f} | {sub_rec.get(10, 0.0):<10.4f} | "
            f"{sub_mrr:<8.4f}"
        )
        print(line_str)

    # 6. Save benchmark summary artifact
    summary = {
        "benchmark_file": "data/benchmarks/eval_public.jsonl",
        "model": provider.model_name,
        "dimension": provider.dimension,
        "total_questions": len(questions),
        "recall_at_1": recalls.get(1, 0.0),
        "recall_at_5": recalls.get(5, 0.0),
        "recall_at_10": recalls.get(10, 0.0),
        "mrr": mrr,
        "queries_with_gold_at_10": total_hit_10,
        "queries_with_gold_top_20": total_hit_any,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "total_duration_s": total_bench_duration,
        "query_results": query_results,
    }
    out_dir = repo_root / "experiments" / "runs"
    out_path = out_dir / "tigergraph_nomic_public_benchmark.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifact saved to: {out_path}")


if __name__ == "__main__":
    main()
