"""Public benchmark evaluation for Stage 4 A4 Agentic GraphRAG on TigerGraph.

Evaluates 100 public benchmark questions against OlympicGraphRAG.
Does NOT access eval_hidden.jsonl.
Compares results against A0 vector RAG and A1 Adaptive GraphRAG baselines.
Saves results to experiments/runs/agentic_graphrag_public_benchmark.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root and src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from scripts.ingest_graph_entities import get_tigergraph_connection  # noqa: E402
from tgh.embeddings.metrics import (  # noqa: E402
    compute_first_gold_rank,
    compute_mrr,
    compute_recalls,
    is_hit_at_k,
)
from tgh.policies.agentic_orchestrator import (  # noqa: E402
    AgenticGraphRAGOrchestrator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_baseline(path: Path) -> dict[str, Any] | None:
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load baseline {path}: {e}")
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    public_file = repo_root / "data" / "benchmarks" / "eval_public.jsonl"
    a0_file = (
        repo_root / "experiments" / "runs" / "tigergraph_nomic_public_benchmark.json"
    )
    a1_file = (
        repo_root
        / "experiments"
        / "runs"
        / "tigergraph_adaptive_graphrag_public_benchmark.json"
    )
    a4_prev_file = (
        repo_root
        / "experiments"
        / "runs"
        / "agentic_graphrag_previous_public_benchmark.json"
    )

    print("=" * 80)
    print("STAGE 4: FULL PUBLIC AGENTIC GRAPHRAG BENCHMARK (100 QUESTIONS)")
    print("=" * 80)

    # 1. Connect to TigerGraph
    conn = get_tigergraph_connection()
    logger.info(f"Connected to TigerGraph graph: {conn.graphname}")

    # 2. Initialize Agentic Orchestrator
    orchestrator = AgenticGraphRAGOrchestrator(conn=conn)
    logger.info("AgenticGraphRAGOrchestrator initialized with A0, A1, A2, and A3.")

    # 3. Load 100 Public Benchmark Questions
    questions: list[dict[str, Any]] = []
    with open(public_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    logger.info(
        f"Loaded {len(questions)} public benchmark questions from {public_file}"
    )

    # 4. Evaluate each question
    query_results: list[dict[str, Any]] = []
    retrieved_docs_list: list[list[str]] = []
    gold_docs_list: list[list[str]] = []
    latencies_ms: list[float] = []

    strategy_counts: dict[str, int] = {}
    skill_counts: dict[str, int] = {}
    verif_counts: dict[str, int] = {}
    repairs_attempted = 0
    repairs_recovered = 0

    t_bench0 = time.perf_counter()

    for idx, q_item in enumerate(questions, start=1):
        qid = q_item["qid"]
        q_text = q_item["question"]
        qtype = q_item.get("qtype", "unknown")
        gold_docs = q_item.get("gold_doc_ids", [])
        gold_docs_list.append(gold_docs)

        orch_res = orchestrator.run(q_text)

        ranked_docs = orch_res.ranked_doc_ids
        retrieved_docs_list.append(ranked_docs)
        latencies_ms.append(orch_res.latency_ms)

        strat_str = orch_res.strategy.value
        strategy_counts[strat_str] = strategy_counts.get(strat_str, 0) + 1

        for sk in orch_res.skills_invoked:
            skill_counts[sk] = skill_counts.get(sk, 0) + 1

        v_status = (
            orch_res.verification.status.value
            if orch_res.verification
            else "UNVERIFIED"
        )
        verif_counts[v_status] = verif_counts.get(v_status, 0) + 1

        if orch_res.repair_used:
            repairs_attempted += 1
            if any(d in gold_docs for d in ranked_docs[:10]):
                repairs_recovered += 1

        first_rank = compute_first_gold_rank(ranked_docs, gold_docs)
        hit_1 = is_hit_at_k(ranked_docs, gold_docs, 1)
        hit_5 = is_hit_at_k(ranked_docs, gold_docs, 5)
        hit_10 = is_hit_at_k(ranked_docs, gold_docs, 10)
        hit_20 = is_hit_at_k(ranked_docs, gold_docs, 20)

        query_results.append(
            {
                "qid": qid,
                "qtype": qtype,
                "strategy": strat_str,
                "skills_invoked": orch_res.skills_invoked,
                "rank": first_rank,
                "hit_1": hit_1,
                "hit_5": hit_5,
                "hit_10": hit_10,
                "hit_20": hit_20,
                "verification_status": v_status,
                "repair_used": orch_res.repair_used,
                "candidate_answer": orch_res.candidate_answer,
                "latency_ms": round(orch_res.latency_ms, 2),
                "mcp_tools": orch_res.mcp_tools_invoked,
                "a2_tools": orch_res.a2_tools_invoked,
                "evidence_count": len(orch_res.evidence_items),
                "run_id": orch_res.run_id,
                "model_name": orch_res.model_name,
                "input_tokens": orch_res.input_tokens,
                "output_tokens": orch_res.output_tokens,
                "total_tokens": orch_res.total_tokens,
                "model_latency_ms": round(orch_res.model_latency_ms, 2),
            }
        )

        if idx % 10 == 0 or idx == len(questions):
            curr_h1 = sum(1 for r in query_results if r["hit_1"])
            curr_h10 = sum(1 for r in query_results if r["hit_10"])
            print(
                f"  [{idx:3d}/100] | Hit@1: {curr_h1}/{idx} ({curr_h1/idx*100:.1f}%) | "
                f"Hit@10: {curr_h10}/{idx} ({curr_h10/idx*100:.1f}%) | "
                f"Strat: {strat_str} | Latency: {orch_res.latency_ms:.1f}ms"
            )

    total_bench_duration = time.perf_counter() - t_bench0

    # 5. Compute Benchmark Metrics
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

    r1_val = recalls.get(1, 0.0)
    r5_val = recalls.get(5, 0.0)
    r10_val = recalls.get(10, 0.0)
    r20_val = recalls.get(20, 0.0)

    # 6. Question Type Breakdown
    types = sorted({q["qtype"] for q in questions})
    qtype_metrics: dict[str, Any] = {}
    for qt in types:
        sub_indices = [i for i, q in enumerate(questions) if q["qtype"] == qt]
        sub_ret = [retrieved_docs_list[i] for i in sub_indices]
        sub_gold = [gold_docs_list[i] for i in sub_indices]
        sub_rec = compute_recalls(sub_ret, sub_gold, k_list=(1, 5, 10, 20))
        sub_mrr = compute_mrr(sub_ret, sub_gold)
        sub_lats = [latencies_ms[i] for i in sub_indices]

        qtype_metrics[qt] = {
            "count": len(sub_indices),
            "recall_at_1": sub_rec.get(1, 0.0),
            "recall_at_5": sub_rec.get(5, 0.0),
            "recall_at_10": sub_rec.get(10, 0.0),
            "recall_at_20": sub_rec.get(20, 0.0),
            "mrr": sub_mrr,
            "avg_latency_ms": float(np.mean(sub_lats)),
        }

    # Load Baselines for Comparison
    a0_bench = load_baseline(a0_file)
    a1_bench = load_baseline(a1_file)
    a4_prev = load_baseline(a4_prev_file)

    print("\n" + "=" * 80)
    print("FINAL A4 AGENTIC GRAPHRAG PUBLIC BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Questions Evaluated:    {len(questions)}")
    print(f"Recall@1:                     {r1_val:.4f} ({r1_val * 100:.2f}%)")
    print(f"Recall@5:                     {r5_val:.4f} ({r5_val * 100:.2f}%)")
    print(f"Recall@10:                    {r10_val:.4f} ({r10_val * 100:.2f}%)")
    print(f"Recall@20:                    {r20_val:.4f} ({r20_val * 100:.2f}%)")
    print(f"Mean Reciprocal Rank (MRR):   {mrr:.4f}")
    print(f"Queries with Gold Hit (@1):   {total_hit_1} / {len(questions)}")
    print(f"Queries with Gold Hit (@5):   {total_hit_5} / {len(questions)}")
    print(f"Queries with Gold Hit (@10):  {total_hit_10} / {len(questions)}")
    print(f"Queries with Gold Hit (@20):  {total_hit_20} / {len(questions)}")
    print("\nDeepAgents Skills Breakdown:")
    for sk_name, cnt in sorted(skill_counts.items(), key=lambda x: -x[1]):
        print(f"  {sk_name:<30}: {cnt:3d} ({cnt/len(questions)*100:.1f}%)")
    print("\nStrategy Breakdown:")
    for strat, cnt in sorted(strategy_counts.items()):
        print(f"  {strat:<28}: {cnt:3d} ({cnt/len(questions)*100:.1f}%)")
    print("\nVerification & Repair:")
    for v_st, cnt in sorted(verif_counts.items()):
        print(f"  {v_st:<28}: {cnt:3d} ({cnt/len(questions)*100:.1f}%)")
    print(f"  Repairs Attempted:           {repairs_attempted}")
    print(f"  Repairs Recovered (@10):     {repairs_recovered}")
    print("\nLatency Statistics (End-to-End per Query):")
    print(f"  Average Latency:            {avg_latency:.2f} ms")
    print(f"  p50 (Median) Latency:       {p50_latency:.2f} ms")
    print(f"  p95 Latency:                {p95_latency:.2f} ms")
    print(f"  p99 Latency:                {p99_latency:.2f} ms")
    print(f"  Total Benchmark Duration:   {total_bench_duration:.2f} s")

    print("\n--- Performance by Question Type ---")
    header = (
        f"{'Question Type':<16} | {'Count':<5} | {'Recall@1':<9} | "
        f"{'Recall@5':<9} | {'Recall@10':<9} | {'Recall@20':<9} | {'MRR':<8}"
    )
    print(header)
    print("-" * 85)
    for qt, m in qtype_metrics.items():
        print(
            f"{qt:<16} | {m['count']:<5} | "
            f"{m['recall_at_1'] * 100:>8.2f}% | "
            f"{m['recall_at_5'] * 100:>8.2f}% | "
            f"{m['recall_at_10'] * 100:>8.2f}% | "
            f"{m['recall_at_20'] * 100:>8.2f}% | "
            f"{m['mrr']:>7.4f}"
        )

    # 7. Baseline Comparison Table
    print("\n" + "=" * 80)
    print("COMPARISON: A0 (Vector) vs A1 (Adaptive) vs A4 Previous vs A4 Skill-Enabled")
    print("=" * 80)
    comp_header = (
        f"{'Pipeline':<24} | {'Recall@1':<9} | {'Recall@5':<9} | "
        f"{'Recall@10':<9} | {'Recall@20':<9} | {'MRR':<8} | {'Avg Latency':<12}"
    )
    print(comp_header)
    print("-" * 92)

    if a0_bench:
        a0_m = a0_bench.get("metrics") or a0_bench
        r20_a0 = a0_m.get("recall_at_20") or (
            a0_m.get("queries_with_gold_top_20", 0) / 100
        )
        print(
            f"{'A0 Vector Baseline':<24} | "
            f"{a0_m.get('recall_at_1', 0)*100:>8.2f}% | "
            f"{a0_m.get('recall_at_5', 0)*100:>8.2f}% | "
            f"{a0_m.get('recall_at_10', 0)*100:>8.2f}% | "
            f"{r20_a0 * 100:>8.2f}% | "
            f"{a0_m.get('mrr', 0):>7.4f} | "
            f"{a0_m.get('avg_latency_ms', 0):>9.2f} ms"
        )
    if a1_bench:
        a1_m = (
            a1_bench.get("overall_metrics")
            or a1_bench.get("metrics")
            or a1_bench
        )
        print(
            f"{'A1 Adaptive GraphRAG':<24} | "
            f"{a1_m.get('recall_at_1', 0)*100:>8.2f}% | "
            f"{a1_m.get('recall_at_5', 0)*100:>8.2f}% | "
            f"{a1_m.get('recall_at_10', 0)*100:>8.2f}% | "
            f"{a1_m.get('recall_at_20', 0)*100:>8.2f}% | "
            f"{a1_m.get('mrr', 0):>7.4f} | "
            f"{a1_bench.get('latency_ms', {}).get('mean', 2370.0):>9.2f} ms"
        )
    if a4_prev:
        a4p_m = a4_prev.get("metrics") or a4_prev
        print(
            f"{'A4 Previous (No Skills)':<24} | "
            f"{a4p_m.get('recall_at_1', 0)*100:>8.2f}% | "
            f"{a4p_m.get('recall_at_5', 0)*100:>8.2f}% | "
            f"{a4p_m.get('recall_at_10', 0)*100:>8.2f}% | "
            f"{a4p_m.get('recall_at_20', 0)*100:>8.2f}% | "
            f"{a4p_m.get('mrr', 0):>7.4f} | "
            f"{a4p_m.get('avg_latency_ms', 0):>9.2f} ms"
        )
    print(
        f"{'A4 Skill-Enabled (Now)':<24} | "
        f"{r1_val * 100:>8.2f}% | "
        f"{r5_val * 100:>8.2f}% | "
        f"{r10_val * 100:>8.2f}% | "
        f"{r20_val * 100:>8.2f}% | "
        f"{mrr:>7.4f} | "
        f"{avg_latency:>9.2f} ms"
    )

    # 8. Save Full Artifact
    out_dir = repo_root / "experiments" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agentic_graphrag_public_benchmark.json"

    benchmark_record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "benchmark_file": str(public_file),
        "total_queries": len(questions),
        "pipeline": "A4_Agentic_GraphRAG_Skills",
        "metrics": {
            "recall_at_1": r1_val,
            "recall_at_5": r5_val,
            "recall_at_10": r10_val,
            "recall_at_20": r20_val,
            "mrr": mrr,
            "hit_at_1_count": total_hit_1,
            "hit_at_5_count": total_hit_5,
            "hit_at_10_count": total_hit_10,
            "hit_at_20_count": total_hit_20,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_benchmark_duration_s": total_bench_duration,
        },
        "skill_breakdown": skill_counts,
        "strategy_breakdown": strategy_counts,
        "verification_breakdown": verif_counts,
        "repairs": {
            "attempted": repairs_attempted,
            "recovered_at_10": repairs_recovered,
        },
        "qtype_breakdown": qtype_metrics,
        "model_telemetry": {
            "model_name": orchestrator.model_name,
            "total_input_tokens": sum(
                r.get("input_tokens", 0) for r in query_results
            ),
            "total_output_tokens": sum(
                r.get("output_tokens", 0) for r in query_results
            ),
            "total_tokens": sum(r.get("total_tokens", 0) for r in query_results),
        },
        "query_results": query_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2)

    logger.info(f"\nArtifact saved to: {out_path}")


if __name__ == "__main__":
    main()
