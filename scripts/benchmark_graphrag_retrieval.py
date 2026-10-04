"""Public benchmark evaluation using deterministic GraphRAG retrieval on TigerGraph.

Evaluates 100 public benchmark questions against OlympicGraphRAG.
Does NOT access eval_hidden.jsonl.
Compares results against the vector-only baseline.
"""

import json
import time
from datetime import UTC, datetime
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
from tgh.retrieval.graph import GraphExpansionConfig
from tgh.retrieval.graphrag import GraphRAGRetriever


def load_env() -> dict[str, str]:
    env_file = Path("/Users/yeshi/Desktop/tgh/.env")
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
    repo_root = Path("/Users/yeshi/Desktop/tgh")
    public_file = repo_root / "data" / "benchmarks" / "eval_public.jsonl"
    baseline_file = (
        repo_root / "experiments" / "runs" / "tigergraph_nomic_public_benchmark.json"
    )

    print("=" * 80)
    print("STAGE 2: PUBLIC ADAPTIVE GRAPHRAG BENCHMARK ON TIGERGRAPH (100 QUESTIONS)")
    print("=" * 80)

    # 1. Connect to TigerGraph
    env = load_env()
    conn = tg.TigerGraphConnection(
        host=env["TIGERGRAPH_HOST"],
        graphname=env["TIGERGRAPH_GRAPH_NAME"],
        gsqlSecret=env["TIGERGRAPH_SECRET"],
    )
    conn.getToken()
    print(f"Connected to graph: {conn.graphname}")

    # 2. Initialize Nomic Embedding Provider & GraphRAG Retriever
    t_m0 = time.perf_counter()
    provider = NomicEmbeddingProvider(dimension=768)
    config = GraphExpansionConfig(
        seed_top_k=20,
        max_seeds_to_expand=5,
        max_entities_per_seed=5,
        max_domain_vertices_per_seed=5,
        max_events_per_person=5,
        max_events_per_venue=10,
        max_graph_derived_chunks=20,
        max_hops=2,
        decay_per_hop=0.90,
        reinforcement_weight=0.05,
        adaptive_gating=True,
        min_vector_confidence=0.72,
        min_vector_margin=0.015,
        num_workers=6,
    )
    retriever = GraphRAGRetriever(
        conn=conn,
        embedding_provider=provider,
        config=config,
    )
    print(f"Loaded {provider.model_name} in {time.perf_counter() - t_m0:.2f}s")

    # 3. Load 100 public questions
    with public_file.open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} public evaluation questions.")

    retrieved_docs_list = []
    gold_docs_list = []
    latencies_ms = []
    query_results = []

    print("\n--- Running 100-Query GraphRAG Benchmark ---")
    t_bench0 = time.perf_counter()

    for idx, q in enumerate(questions, start=1):
        qid = q["qid"]
        qtype = q["qtype"]
        question = q["question"]
        gold_set = set(q["gold_doc_ids"])
        gold_docs_list.append(gold_set)

        res = retriever.retrieve(question)

        ranked_docs = res.ranked_doc_ids
        retrieved_docs_list.append(ranked_docs)
        latencies_ms.append(res.latency_ms)

        first_rank = compute_first_gold_rank(ranked_docs, gold_set)
        hit_1 = is_hit_at_k(ranked_docs, gold_set, 1)
        hit_5 = is_hit_at_k(ranked_docs, gold_set, 5)
        hit_10 = is_hit_at_k(ranked_docs, gold_set, 10)
        hit_20 = is_hit_at_k(ranked_docs, gold_set, 20)

        # Record provenance sample for reporting
        prov_sample = [
            ec.provenance[0] for ec in res.evidence_chunks[:3] if ec.provenance
        ]

        query_results.append(
            {
                "qid": qid,
                "qtype": qtype,
                "rank": first_rank,
                "hit_1": hit_1,
                "hit_5": hit_5,
                "hit_10": hit_10,
                "hit_20": hit_20,
                "latency_ms": res.latency_ms,
                "is_graph_expanded": res.is_graph_expanded,
                "decision_reason": res.decision_reason,
                "num_evidence_chunks": len(res.evidence_chunks),
                "num_graph_derived": res.expansion_stats.get(
                    "num_graph_derived_chunks", 0
                ),
                "provenance_sample": prov_sample,
            }
        )

        if idx % 10 == 0 or idx == len(questions):
            curr_hits_10 = sum(1 for r in query_results if r["hit_10"])
            curr_hits_1 = sum(1 for r in query_results if r["hit_1"])
            pct_1 = curr_hits_1 / idx * 100
            pct_10 = curr_hits_10 / idx * 100
            print(
                f"  [{idx:3d}/100] | Hit@1: {curr_hits_1}/{idx} ({pct_1:.1f}%) | "
                f"Hit@10: {curr_hits_10}/{idx} ({pct_10:.1f}%) | "
                f"Latency: {res.latency_ms:.1f}ms"
            )

    total_bench_duration = time.perf_counter() - t_bench0

    # 4. Compute Benchmark Metrics
    recalls = compute_recalls(
        retrieved_docs_list, gold_docs_list, k_list=(1, 5, 10, 20)
    )
    mrr = compute_mrr(retrieved_docs_list, gold_docs_list)

    avg_latency = float(np.mean(latencies_ms))
    p50_latency = float(np.percentile(latencies_ms, 50))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))

    total_hit_1 = sum(1 for r in query_results if r["hit_1"])
    total_hit_5 = sum(1 for r in query_results if r["hit_5"])
    total_hit_10 = sum(1 for r in query_results if r["hit_10"])
    total_hit_20 = sum(1 for r in query_results if r["hit_20"])

    print("\n" + "=" * 80)
    print("FINAL PUBLIC BENCHMARK RESULTS (GRAPHRAG DETERMINISTIC RETRIEVAL)")
    print("=" * 80)
    print(f"Total Questions Evaluated:    {len(questions)}")
    r1_val = recalls.get(1, 0.0)
    r5_val = recalls.get(5, 0.0)
    r10_val = recalls.get(10, 0.0)
    r20_val = recalls.get(20, 0.0)
    print(f"Recall@1:                     {r1_val:.4f} ({r1_val * 100:.2f}%)")
    print(f"Recall@5:                     {r5_val:.4f} ({r5_val * 100:.2f}%)")
    print(f"Recall@10:                    {r10_val:.4f} ({r10_val * 100:.2f}%)")
    print(f"Recall@20:                    {r20_val:.4f} ({r20_val * 100:.2f}%)")
    print(f"Mean Reciprocal Rank (MRR):   {mrr:.4f}")
    print(f"Queries with Gold Hit (@1):   {total_hit_1} / {len(questions)}")
    print(f"Queries with Gold Hit (@5):   {total_hit_5} / {len(questions)}")
    print(f"Queries with Gold Hit (@10):  {total_hit_10} / {len(questions)}")
    print(f"Queries with Gold Hit (@20):  {total_hit_20} / {len(questions)}")
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
        f"{'Question Type':<16} | {'Count':<5} | {'Recall@1':<9} | "
        f"{'Recall@5':<9} | {'Recall@10':<9} | {'Recall@20':<9} | {'MRR':<8}"
    )
    print(header)
    print("-" * 85)

    qtype_metrics = {}
    for qt in types:
        sub_indices = [i for i, q in enumerate(questions) if q["qtype"] == qt]
        sub_ret = [retrieved_docs_list[i] for i in sub_indices]
        sub_gold = [gold_docs_list[i] for i in sub_indices]
        sub_rec = compute_recalls(sub_ret, sub_gold, k_list=(1, 5, 10, 20))
        sub_mrr = compute_mrr(sub_ret, sub_gold)

        qtype_metrics[qt] = {
            "count": len(sub_indices),
            "recall_at_1": sub_rec.get(1, 0.0),
            "recall_at_5": sub_rec.get(5, 0.0),
            "recall_at_10": sub_rec.get(10, 0.0),
            "recall_at_20": sub_rec.get(20, 0.0),
            "mrr": sub_mrr,
        }

        line_str = (
            f"{qt:<16} | {len(sub_indices):<5} | {sub_rec.get(1, 0.0):<9.4f} | "
            f"{sub_rec.get(5, 0.0):<9.4f} | {sub_rec.get(10, 0.0):<9.4f} | "
            f"{sub_rec.get(20, 0.0):<9.4f} | {sub_mrr:<8.4f}"
        )
        print(line_str)

    # 6. Comparison against baseline
    if baseline_file.is_file():
        base_data = json.loads(baseline_file.read_text(encoding="utf-8"))
        print("\n" + "=" * 80)
        print("COMPARISON: GRAPHRAG VS VECTOR-ONLY BASELINE")
        print("=" * 80)
        b_r1 = base_data.get("recall_at_1", 0.0)
        b_r5 = base_data.get("recall_at_5", 0.0)
        b_r10 = base_data.get("recall_at_10", 0.0)
        b_mrr = base_data.get("mrr", 0.0)

        r1 = recalls.get(1, 0.0)
        r5 = recalls.get(5, 0.0)
        r10 = recalls.get(10, 0.0)
        print(
            f"Overall Recall@1:   Vector={b_r1:.4f} -> "
            f"GraphRAG={r1:.4f} ({r1 - b_r1:+.4f})"
        )
        print(
            f"Overall Recall@5:   Vector={b_r5:.4f} -> "
            f"GraphRAG={r5:.4f} ({r5 - b_r5:+.4f})"
        )
        print(
            f"Overall Recall@10:  Vector={b_r10:.4f} -> "
            f"GraphRAG={r10:.4f} ({r10 - b_r10:+.4f})"
        )
        print(
            f"Overall MRR:        Vector={b_mrr:.4f} -> "
            f"GraphRAG={mrr:.4f} ({mrr - b_mrr:+.4f})"
        )

        # Multi-hop comparison
        base_mh_hits_10 = sum(
            1
            for r in base_data.get("query_results", [])
            if r["qtype"] == "multi_hop" and r.get("hit_10")
        )
        base_mh_hits_1 = sum(
            1
            for r in base_data.get("query_results", [])
            if r["qtype"] == "multi_hop" and r.get("hit_1")
        )
        base_mh_count = sum(
            1 for r in base_data.get("query_results", []) if r["qtype"] == "multi_hop"
        )
        if base_mh_count > 0:
            bm_r1 = base_mh_hits_1 / base_mh_count
            bm_r10 = base_mh_hits_10 / base_mh_count
            gm_r1 = qtype_metrics.get("multi_hop", {}).get("recall_at_1", 0.0)
            gm_r10 = qtype_metrics.get("multi_hop", {}).get("recall_at_10", 0.0)
            gm_mrr = qtype_metrics.get("multi_hop", {}).get("mrr", 0.0)
            print("\nMULTI-HOP BREAKDOWN:")
            d_r1 = gm_r1 - bm_r1
            d_r10 = gm_r10 - bm_r10
            print(
                f"  Multi-Hop Recall@1:  Vector={bm_r1:.4f} -> "
                f"GraphRAG={gm_r1:.4f} ({d_r1:+.4f})"
            )
            print(
                f"  Multi-Hop Recall@10: Vector={bm_r10:.4f} -> "
                f"GraphRAG={gm_r10:.4f} ({d_r10:+.4f})"
            )
            print(f"  Multi-Hop MRR:       Vector=0.2408 -> GraphRAG={gm_mrr:.4f}")

    # 7. Save artifact
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "strategy_name": "GraphRAG_Adaptive_v1",
        "benchmark_file": "data/benchmarks/eval_public.jsonl",
        "embedding_model": provider.model_name,
        "embedding_dimension": provider.dimension,
        "retrieval_configuration": {
            "seed_top_k": config.seed_top_k,
            "max_seeds_to_expand": config.max_seeds_to_expand,
            "max_entities_per_seed": config.max_entities_per_seed,
            "max_domain_vertices_per_seed": config.max_domain_vertices_per_seed,
            "max_events_per_person": config.max_events_per_person,
            "max_events_per_venue": config.max_events_per_venue,
            "max_graph_derived_chunks": config.max_graph_derived_chunks,
            "max_hops": config.max_hops,
            "decay_per_hop": config.decay_per_hop,
            "reinforcement_weight": config.reinforcement_weight,
            "adaptive_gating": config.adaptive_gating,
            "min_vector_confidence": config.min_vector_confidence,
            "min_vector_margin": config.min_vector_margin,
        },
        "total_questions": len(questions),
        "overall_metrics": {
            "recall_at_1": recalls.get(1, 0.0),
            "recall_at_5": recalls.get(5, 0.0),
            "recall_at_10": recalls.get(10, 0.0),
            "recall_at_20": recalls.get(20, 0.0),
            "mrr": mrr,
            "queries_with_gold_at_1": total_hit_1,
            "queries_with_gold_at_5": total_hit_5,
            "queries_with_gold_at_10": total_hit_10,
            "queries_with_gold_at_20": total_hit_20,
        },
        "qtype_metrics": qtype_metrics,
        "latency_metrics": {
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_benchmark_duration_s": total_bench_duration,
        },
        "query_results": query_results,
    }

    out_dir = repo_root / "experiments" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tigergraph_adaptive_graphrag_public_benchmark.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nArtifact saved to: {out_path}")


if __name__ == "__main__":
    main()
