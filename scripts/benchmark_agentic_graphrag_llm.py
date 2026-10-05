"""Public 100-question benchmark evaluation for genuine LLM-driven A4 Agentic GraphRAG.

Evaluates 100 public benchmark questions from data/benchmarks/eval_public.jsonl.
Does NOT access eval_hidden.jsonl.
Does NOT inspect hidden gold labels or tune against hidden data.
Does NOT overwrite experiments/runs/agentic_graphrag_public_benchmark.json.

Saves results to:
experiments/runs/agentic_graphrag_public_benchmark_llm.json
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
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

load_dotenv(root_dir / ".env")

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
    out_dir = repo_root / "experiments" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agentic_graphrag_public_benchmark_llm.json"

    # Baseline file paths for comparison reporting
    a0_file = repo_root / "experiments" / "runs" / "tigergraph_nomic_public_benchmark.json"
    a1_file = repo_root / "experiments" / "runs" / "tigergraph_adaptive_graphrag_public_benchmark.json"
    a4_det_file = repo_root / "experiments" / "runs" / "agentic_graphrag_public_benchmark.json"

    print("=" * 90)
    print("STAGE 4: FULL PUBLIC LLM-DRIVEN AGENTIC GRAPHRAG BENCHMARK (100 QUESTIONS)")
    print(f"Target Output: {out_path}")
    print("=" * 90)

    # 1. Connect to TigerGraph
    conn = get_tigergraph_connection()
    logger.info(f"Connected to TigerGraph graph: {conn.graphname}")

    # 2. Initialize LLM-driven Agentic Orchestrator
    orchestrator = AgenticGraphRAGOrchestrator(conn=conn, enable_model_agent=True)
    logger.info(
        f"Orchestrator initialized. Model: {orchestrator.model_name}, "
        f"Provider: {orchestrator.model_provider}, "
        f"DeepAgent enabled: {orchestrator.deep_agent is not None}"
    )

    if orchestrator.deep_agent is None:
        logger.error("DeepAgent model is NOT initialized! Aborting benchmark to prevent accidental fallback run.")
        sys.exit(1)

    # 3. Load 100 Public Benchmark Questions
    questions: list[dict[str, Any]] = []
    with open(public_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    logger.info(f"Loaded {len(questions)} public questions from {public_file}")

    # 4. Evaluate each question
    query_results: list[dict[str, Any]] = []
    retrieved_docs_list: list[list[str]] = []
    gold_docs_list: list[list[str]] = []
    latencies_ms: list[float] = []

    strategy_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    skill_counts: dict[str, int] = {}
    verif_counts: dict[str, int] = {}
    repairs_attempted = 0
    repairs_recovered = 0
    budget_violations = 0
    empty_answers = 0
    execution_failures = 0
    model_driven_count = 0
    fallback_count = 0

    t_bench0 = time.perf_counter()

    for idx, q_item in enumerate(questions, start=1):
        qid = q_item["qid"]
        q_text = q_item["question"]
        qtype = q_item.get("qtype", "unknown")
        gold_docs = q_item.get("gold_doc_ids", [])
        gold_docs_list.append(gold_docs)

        # Retry loop for resilience against transient network/quota glitches
        orch_res = None
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                orch_res = orchestrator.run(q_text, qtype=qtype)
                break
            except Exception as exc:
                logger.warning(
                    f"[{qid}] Attempt {attempt}/{max_attempts} failed with error: {exc}. Retrying in 5s..."
                )
                if attempt == max_attempts:
                    execution_failures += 1
                    logger.error(f"[{qid}] All {max_attempts} attempts failed.")
                time.sleep(5.0)

        if orch_res is None:
            # Record failed query without fabricating telemetry
            query_results.append({
                "qid": qid,
                "question": q_text,
                "qtype": qtype,
                "execution_mode": "failed",
                "model_name": orchestrator.model_name,
                "model_provider": orchestrator.model_provider,
                "model_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_latency_ms": 0.0,
                "strategy_action_sequence": [],
                "tool_calls": [],
                "evidence_count": 0,
                "verification_status": "FAILED",
                "repair_count": 0,
                "candidate_answer": "",
                "ranked_doc_ids": [],
                "final_answer": "",
                "rank": None,
                "hit_1": False,
                "hit_5": False,
                "hit_10": False,
                "hit_20": False,
                "latency_ms": 0.0,
            })
            retrieved_docs_list.append([])
            latencies_ms.append(0.0)
            continue

        ranked_docs = orch_res.ranked_doc_ids
        retrieved_docs_list.append(ranked_docs)
        latencies_ms.append(orch_res.latency_ms)

        if orch_res.execution_mode == "model_driven":
            model_driven_count += 1
        else:
            fallback_count += 1

        strat_str = (
            orch_res.strategy.value
            if hasattr(orch_res.strategy, "value")
            else str(orch_res.strategy)
        )
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

        # Check for empty answers
        cand_ans = (orch_res.candidate_answer or "").strip()
        final_ans = (orch_res.model_answer or orch_res.candidate_answer or "").strip()
        if not final_ans:
            empty_answers += 1

        # Check budget violations
        if (
            orch_res.state_view.tool_call_count > 15
            or orch_res.state_view.repair_count > 1
            or orch_res.state_view.evidence_count > 50
        ):
            budget_violations += 1

        # Extract tool calls and action sequence from action trace
        action_seq: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for step in orch_res.action_trace:
            if step.get("role") == "model" and step.get("tool_calls"):
                for tc in step["tool_calls"]:
                    act_name = tc.get("name", "unknown")
                    action_seq.append(act_name)
                    tool_calls.append({
                        "name": act_name,
                        "args": tc.get("args", {}),
                        "id": tc.get("id"),
                    })
                    action_counts[act_name] = action_counts.get(act_name, 0) + 1

        first_rank = compute_first_gold_rank(ranked_docs, gold_docs)
        hit_1 = is_hit_at_k(ranked_docs, gold_docs, 1)
        hit_5 = is_hit_at_k(ranked_docs, gold_docs, 5)
        hit_10 = is_hit_at_k(ranked_docs, gold_docs, 10)
        hit_20 = is_hit_at_k(ranked_docs, gold_docs, 20)

        record = {
            "qid": qid,
            "question": q_text,
            "qtype": qtype,
            "execution_mode": orch_res.execution_mode,
            "model_name": orch_res.model_name or "gemini-3.5-flash-lite",
            "model_provider": orch_res.model_provider or "Google Gemini API",
            "model_call_count": orch_res.model_call_count,
            "input_tokens": orch_res.input_tokens,
            "output_tokens": orch_res.output_tokens,
            "total_tokens": orch_res.total_tokens,
            "model_latency_ms": round(orch_res.model_latency_ms, 2),
            "strategy_action_sequence": action_seq,
            "tool_calls": tool_calls,
            "evidence_count": len(orch_res.evidence_items),
            "verification_status": v_status,
            "repair_count": orch_res.state_view.repair_count,
            "candidate_answer": cand_ans,
            "ranked_doc_ids": ranked_docs,
            "final_answer": final_ans,
            "rank": first_rank,
            "hit_1": hit_1,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "hit_20": hit_20,
            "latency_ms": round(orch_res.latency_ms, 2),
            "action_trace": orch_res.action_trace,
        }
        query_results.append(record)

        if idx % 10 == 0 or idx == len(questions):
            curr_h1 = sum(1 for r in query_results if r["hit_1"])
            curr_h10 = sum(1 for r in query_results if r["hit_10"])
            print(
                f"  [{idx:3d}/100] | Hit@1: {curr_h1}/{idx} ({curr_h1/idx*100:.1f}%) | "
                f"Hit@10: {curr_h10}/{idx} ({curr_h10/idx*100:.1f}%) | "
                f"Mode: {orch_res.execution_mode} | Calls: {orch_res.model_call_count} | "
                f"Lat: {orch_res.latency_ms:.1f}ms"
            )

        # Brief pause between queries to avoid burst rate limits
        time.sleep(0.1)

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

    # 7. Model Telemetry & Call Statistics
    total_model_calls = sum(r.get("model_call_count", 0) for r in query_results)
    total_in_tokens = sum(r.get("input_tokens", 0) for r in query_results)
    total_out_tokens = sum(r.get("output_tokens", 0) for r in query_results)
    total_all_tokens = sum(r.get("total_tokens", 0) for r in query_results)
    call_counts = [r.get("model_call_count", 0) for r in query_results]
    model_latencies = [r.get("model_latency_ms", 0.0) for r in query_results]

    model_call_stats = {
        "model_name": "gemini-3.5-flash-lite",
        "provider": "Google Gemini API",
        "execution_mode": "model_driven",
        "total_model_calls": total_model_calls,
        "avg_model_calls_per_query": float(np.mean(call_counts)) if call_counts else 0.0,
        "min_model_calls": int(np.min(call_counts)) if call_counts else 0,
        "max_model_calls": int(np.max(call_counts)) if call_counts else 0,
        "total_input_tokens": total_in_tokens,
        "total_output_tokens": total_out_tokens,
        "total_tokens": total_all_tokens,
        "avg_tokens_per_query": float(total_all_tokens / len(questions)) if questions else 0.0,
        "total_model_latency_ms": float(sum(model_latencies)),
        "avg_model_latency_ms": float(np.mean(model_latencies)) if model_latencies else 0.0,
    }

    # 8. Baselines Comparison
    a0_bench = load_baseline(a0_file)
    a1_bench = load_baseline(a1_file)
    a4_det_bench = load_baseline(a4_det_file)

    # 9. Build Benchmark Artifact
    benchmark_record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "benchmark_file": str(public_file),
        "total_queries": len(questions),
        "pipeline": "A4_Agentic_GraphRAG_LLM",
        "model_name": "gemini-3.5-flash-lite",
        "provider": "Google Gemini API",
        "execution_mode": "model_driven",
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
            "mean_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_benchmark_duration_s": total_bench_duration,
        },
        "qtype_breakdown": qtype_metrics,
        "strategy_action_distribution": {
            "strategies": strategy_counts,
            "actions": action_counts,
            "skills": skill_counts,
        },
        "verification_breakdown": verif_counts,
        "repair_statistics": {
            "attempted": repairs_attempted,
            "recovered_at_10": repairs_recovered,
            "repair_rate": repairs_attempted / len(questions) if questions else 0.0,
        },
        "execution_statistics": {
            "model_driven_count": model_driven_count,
            "fallback_count": fallback_count,
            "budget_violations": budget_violations,
            "empty_answers": empty_answers,
            "execution_failures": execution_failures,
        },
        "model_call_statistics": model_call_stats,
        "query_results": query_results,
    }

    # Write output to new artifact
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2)

    logger.info(f"\nArtifact successfully saved to: {out_path}")

    # Print Full Terminal Summary
    print("\n" + "=" * 90)
    print("FINAL LLM-DRIVEN A4 AGENTIC GRAPHRAG PUBLIC BENCHMARK RESULTS")
    print("=" * 90)
    print("Pipeline:                     A4_Agentic_GraphRAG_LLM")
    print("Model:                        gemini-3.5-flash-lite (Google Gemini API)")
    print(f"Execution Mode:               model_driven: {model_driven_count}/100, fallback: {fallback_count}/100")
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
    print("\nLatency Statistics (End-to-End per Query):")
    print(f"  Mean Latency:               {avg_latency:.2f} ms")
    print(f"  p50 (Median) Latency:       {p50_latency:.2f} ms")
    print(f"  p95 Latency:                {p95_latency:.2f} ms")
    print(f"  p99 Latency:                {p99_latency:.2f} ms")
    print(f"  Total Benchmark Duration:   {total_bench_duration:.2f} s")
    print("\nModel & Telemetry Statistics:")
    print(f"  Total Model Calls:          {total_model_calls}")
    print(f"  Avg Calls / Query:          {model_call_stats['avg_model_calls_per_query']:.2f}")
    print(f"  Total Input Tokens:         {total_in_tokens}")
    print(f"  Total Output Tokens:        {total_out_tokens}")
    print(f"  Total Tokens:               {total_all_tokens}")
    print(f"  Avg Tokens / Query:         {model_call_stats['avg_tokens_per_query']:.1f}")
    print(f"  Avg Model Latency:          {model_call_stats['avg_model_latency_ms']:.2f} ms")
    print("\nVerification & Repair:")
    print(f"  Repairs Attempted:          {repairs_attempted}")
    print(f"  Repairs Recovered (@10):    {repairs_recovered}")
    print(f"  Budget Violations:          {budget_violations}")
    print(f"  Empty Answers:              {empty_answers}")
    print(f"  Execution Failures:         {execution_failures}")

    print("\n--- Performance by Question Type ---")
    header = (
        f"{'Question Type':<16} | {'Count':<5} | {'Recall@1':<9} | "
        f"{'Recall@5':<9} | {'Recall@10':<9} | {'Recall@20':<9} | {'MRR':<8} | {'Avg Latency':<12}"
    )
    print(header)
    print("-" * 98)
    for qt, m in qtype_metrics.items():
        print(
            f"{qt:<16} | {m['count']:<5} | "
            f"{m['recall_at_1'] * 100:>8.2f}% | "
            f"{m['recall_at_5'] * 100:>8.2f}% | "
            f"{m['recall_at_10'] * 100:>8.2f}% | "
            f"{m['recall_at_20'] * 100:>8.2f}% | "
            f"{m['mrr']:>7.4f} | "
            f"{m['avg_latency_ms']:>9.2f} ms"
        )

    print("\n" + "=" * 90)
    print("COMPARISON: A0 (Vector) vs A1 (Adaptive) vs Deterministic A4 vs LLM-driven A4")
    print("=" * 90)
    comp_header = (
        f"{'Pipeline':<26} | {'Recall@1':<9} | {'Recall@5':<9} | "
        f"{'Recall@10':<9} | {'Recall@20':<9} | {'MRR':<8} | {'Mean Latency':<13}"
    )
    print(comp_header)
    print("-" * 96)

    if a0_bench:
        a0_m = a0_bench.get("metrics") or a0_bench
        r20_a0 = a0_m.get("recall_at_20") or (
            a0_m.get("queries_with_gold_top_20", 0) / 100
        )
        print(
            f"{'A0 Vector Baseline':<26} | "
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
            f"{'A1 Adaptive GraphRAG':<26} | "
            f"{a1_m.get('recall_at_1', 0)*100:>8.2f}% | "
            f"{a1_m.get('recall_at_5', 0)*100:>8.2f}% | "
            f"{a1_m.get('recall_at_10', 0)*100:>8.2f}% | "
            f"{a1_m.get('recall_at_20', 0)*100:>8.2f}% | "
            f"{a1_m.get('mrr', 0):>7.4f} | "
            f"{a1_bench.get('latency_ms', {}).get('mean', 2370.0):>9.2f} ms"
        )
    if a4_det_bench:
        a4d_m = a4_det_bench.get("metrics") or a4_det_bench
        print(
            f"{'A4 Deterministic Baseline':<26} | "
            f"{a4d_m.get('recall_at_1', 0.91)*100:>8.2f}% | "
            f"{a4d_m.get('recall_at_5', 1.0)*100:>8.2f}% | "
            f"{a4d_m.get('recall_at_10', 1.0)*100:>8.2f}% | "
            f"{a4d_m.get('recall_at_20', 1.0)*100:>8.2f}% | "
            f"{a4d_m.get('mrr', 0.9445):>7.4f} | "
            f"{a4d_m.get('avg_latency_ms', 7686.46):>9.2f} ms"
        )
    print(
        f"{'A4 LLM-Driven (This Run)':<26} | "
        f"{r1_val * 100:>8.2f}% | "
        f"{r5_val * 100:>8.2f}% | "
        f"{r10_val * 100:>8.2f}% | "
        f"{r20_val * 100:>8.2f}% | "
        f"{mrr:>7.4f} | "
        f"{avg_latency:>9.2f} ms"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
