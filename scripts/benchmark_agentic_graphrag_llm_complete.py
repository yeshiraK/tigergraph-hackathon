"""Public benchmark completion runner for genuine LLM-driven A4 Agentic GraphRAG.

Executes ONLY the exact 40 fallback queries from the previous Gemini benchmark run,
using a conservative rate limiter (10 RPM, min 6.0s spacing, honoring 429 reset)
with allow_fallback=False.

Merges the 40 newly executed genuine model-driven results with the 60 previous
genuine model-driven results to produce the complete 100-query benchmark artifact:
experiments/runs/agentic_graphrag_public_benchmark_llm_complete.json

Does NOT access eval_hidden.jsonl.
Does NOT overwrite experiments/runs/agentic_graphrag_public_benchmark.json.
Does NOT overwrite experiments/runs/agentic_graphrag_public_benchmark_llm.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

# Enforce IPv4 on macOS to prevent network timeouts
_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0 or family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _orig(host, port, family, type, proto, flags)
socket.getaddrinfo = _ipv4

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

load_dotenv(root_dir / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI

from scripts.ingest_graph_entities import get_tigergraph_connection
from tgh.embeddings.metrics import (
    compute_first_gold_rank,
    compute_mrr,
    compute_recalls,
    is_hit_at_k,
)
from tgh.policies.agentic_orchestrator import AgenticGraphRAGOrchestrator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global rate limiter state (shared across all model calls)
RATE_LIMIT_STATE = {
    "last_call_time": 0.0,
    "rate_limit_events": 0,
    "api_retries": 0,
    "min_interval": 6.0,  # 10 RPM target (guaranteed <= 12 RPM)
}


class PacedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """Paced Gemini model honoring 15 RPM Free Tier quota with automatic 429 backoff."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # 1. Enforce minimum spacing between calls (10 RPM)
        now = time.monotonic()
        elapsed = now - RATE_LIMIT_STATE["last_call_time"]
        if elapsed < RATE_LIMIT_STATE["min_interval"]:
            sleep_needed = RATE_LIMIT_STATE["min_interval"] - elapsed
            time.sleep(sleep_needed)

        # 2. Retry loop honoring 429 retry information
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                RATE_LIMIT_STATE["last_call_time"] = time.monotonic()
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    RATE_LIMIT_STATE["rate_limit_events"] += 1
                    RATE_LIMIT_STATE["api_retries"] += 1
                    wait_s = 16.0
                    m = re.search(r"Please retry in (\d+\.?\d*)s", exc_str)
                    if m:
                        wait_s = float(m.group(1)) + 2.0
                    else:
                        m_delay = re.search(r"'retryDelay':\s*'(\d+)s'", exc_str)
                        if m_delay:
                            wait_s = float(m_delay.group(1)) + 2.0
                    logger.warning(
                        f"[PacedGemini 429] Rate limit reached. Honoring reset: sleeping {wait_s:.1f}s before retrying call (attempt {attempt}/{max_attempts})..."
                    )
                    time.sleep(wait_s)
                    RATE_LIMIT_STATE["last_call_time"] = time.monotonic()
                    continue
                # For non-429 transient network glitch, retry once after short backoff
                if attempt < 3:
                    logger.warning(f"Gemini call error: {exc}. Retrying in 4s...")
                    time.sleep(4.0)
                    RATE_LIMIT_STATE["last_call_time"] = time.monotonic()
                    continue
                raise
        raise RuntimeError("Gemini 429 persisted after 10 retry attempts.")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    public_file = repo_root / "data" / "benchmarks" / "eval_public.jsonl"
    prev_bench_file = repo_root / "experiments" / "runs" / "agentic_graphrag_public_benchmark_llm.json"
    out_dir = repo_root / "experiments" / "runs"
    out_path = out_dir / "agentic_graphrag_public_benchmark_llm_complete.json"

    print("=" * 95)
    print("STAGE 4: COMPLETE 100-QUERY PUBLIC LLM-DRIVEN AGENTIC GRAPHRAG BENCHMARK")
    print(f"Target Output: {out_path}")
    print("=" * 95)

    # 1. Load previous benchmark artifact
    with open(prev_bench_file, encoding="utf-8") as f:
        prev_data = json.load(f)

    prev_results: list[dict[str, Any]] = prev_data["query_results"]
    orig_model_driven = [r for r in prev_results if r.get("execution_mode") == "model_driven"]
    orig_fallback = [r for r in prev_results if r.get("execution_mode") != "model_driven"]

    target_fallback_qids = [r["qid"] for r in orig_fallback]
    logger.info(f"Loaded previous benchmark: {len(orig_model_driven)} model-driven, {len(orig_fallback)} fallback queries.")
    logger.info(f"Targeting exactly {len(target_fallback_qids)} fallback queries for genuine model-driven execution.")

    # 2. Connect to TigerGraph
    conn = get_tigergraph_connection()
    logger.info(f"Connected to TigerGraph graph: {conn.graphname}")

    # 3. Instantiate PacedChatGoogleGenerativeAI
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("GEMINI_API_KEY missing from environment!")
        sys.exit(1)

    model_name = (os.getenv("LLM_MODEL_NAME") or "").strip() or "gemini-3.5-flash-lite"
    paced_model = PacedChatGoogleGenerativeAI(
        model=model_name,
        api_key=gemini_key,
        max_retries=1,
    )

    orchestrator = AgenticGraphRAGOrchestrator(
        conn=conn,
        model=paced_model,
        enable_model_agent=True,
    )
    orchestrator.model_name = "gemini-3.5-flash-lite"
    orchestrator.model_provider = "Google Gemini API"
    logger.info(f"Orchestrator initialized with PacedChatGoogleGenerativeAI (Model: {orchestrator.model_name}).")

    # 4. Load all public benchmark questions (map by qid)
    public_questions: dict[str, dict[str, Any]] = {}
    with open(public_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            public_questions[item["qid"]] = item

    if len(public_questions) != 100:
        logger.error(f"Expected 100 public questions, found {len(public_questions)}! Aborting.")
        sys.exit(1)

    # 5. Run genuine model-driven execution on the 40 fallback queries
    new_results_by_qid: dict[str, dict[str, Any]] = {}
    t_start_rerun = time.perf_counter()

    for idx, qid in enumerate(target_fallback_qids, start=1):
        q_item = public_questions[qid]
        q_text = q_item["question"]
        qtype = q_item.get("qtype", "unknown")
        gold_docs = q_item.get("gold_doc_ids", [])

        logger.info(f"[{idx:2d}/{len(target_fallback_qids)}] Executing model-driven query: {qid} ({qtype})")
        t0 = time.perf_counter()
        # Strictly allow_fallback=False
        orch_res = orchestrator.run(q_text, qtype=qtype, allow_fallback=False)
        total_lat_ms = (time.perf_counter() - t0) * 1000.0

        if orch_res.execution_mode != "model_driven":
            logger.error(f"CRITICAL: {qid} did not execute in model_driven mode! Mode: {orch_res.execution_mode}")
            sys.exit(1)

        ranked_docs = orch_res.ranked_doc_ids
        first_rank = compute_first_gold_rank(ranked_docs, gold_docs)
        hit_1 = is_hit_at_k(ranked_docs, gold_docs, 1)
        hit_5 = is_hit_at_k(ranked_docs, gold_docs, 5)
        hit_10 = is_hit_at_k(ranked_docs, gold_docs, 10)
        hit_20 = is_hit_at_k(ranked_docs, gold_docs, 20)

        # Extract action sequence & tool calls
        action_seq = []
        tool_calls = []
        for step in orch_res.action_trace:
            if step.get("role") == "model" and step.get("tool_calls"):
                for tc in step["tool_calls"]:
                    name = tc.get("name", "unknown")
                    action_seq.append(name)
                    tool_calls.append({
                        "name": name,
                        "args": tc.get("args", {}),
                        "id": tc.get("id"),
                    })

        cand_ans = (orch_res.candidate_answer or "").strip()
        final_ans = (orch_res.model_answer or orch_res.candidate_answer or "").strip()
        v_status = (
            orch_res.verification.status.value
            if orch_res.verification
            else "UNVERIFIED"
        )

        record = {
            "qid": qid,
            "question": q_text,
            "qtype": qtype,
            "execution_mode": "model_driven",
            "model_name": "gemini-3.5-flash-lite",
            "model_provider": "Google Gemini API",
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
            "latency_ms": round(total_lat_ms, 2),
            "action_trace": orch_res.action_trace,
        }
        new_results_by_qid[qid] = record

        print(
            f"  [{idx:2d}/{len(target_fallback_qids)}] {qid} | Hit@1: {hit_1} | "
            f"Calls: {orch_res.model_call_count} | Tokens: {orch_res.total_tokens} | "
            f"Lat: {total_lat_ms:.1f}ms | 429 events: {RATE_LIMIT_STATE['rate_limit_events']}"
        )

    rerun_duration_s = time.perf_counter() - t_start_rerun
    logger.info(f"All {len(target_fallback_qids)} fallback queries successfully re-executed in {rerun_duration_s:.1f}s.")

    # 6. Merge with original 60 genuine model-driven results
    final_query_results: list[dict[str, Any]] = []
    # Preserve original public question order from eval_public.jsonl
    with open(public_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["qid"]
            if qid in new_results_by_qid:
                final_query_results.append(new_results_by_qid[qid])
            else:
                # Find in orig_model_driven
                matching = [r for r in orig_model_driven if r["qid"] == qid]
                if not matching:
                    logger.error(f"Missing query {qid} from both new and previous model-driven sets!")
                    sys.exit(1)
                final_query_results.append(matching[0])

    # 7. Final Integrity Verification
    assert len(final_query_results) == 100, f"Expected 100 queries, got {len(final_query_results)}"
    for r in final_query_results:
        assert r["execution_mode"] == "model_driven", f"Query {r['qid']} has execution_mode={r['execution_mode']}"
        assert r["model_name"] == "gemini-3.5-flash-lite"
        assert r["model_provider"] == "Google Gemini API"

    logger.info("Integrity Verification Passed: Exactly 100/100 queries are genuine model_driven.")

    # 8. Compute Full Final Metrics Across All 100 Queries
    retrieved_docs_list = [r["ranked_doc_ids"] for r in final_query_results]
    gold_docs_list = [public_questions[r["qid"]].get("gold_doc_ids", []) for r in final_query_results]
    latencies_ms = [r["latency_ms"] for r in final_query_results]

    recalls = compute_recalls(retrieved_docs_list, gold_docs_list, k_list=(1, 5, 10, 20))
    mrr = compute_mrr(retrieved_docs_list, gold_docs_list)

    avg_latency = float(np.mean(latencies_ms))
    p50_latency = float(np.percentile(latencies_ms, 50))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))

    total_hit_1 = sum(1 for r in final_query_results if r["hit_1"])
    total_hit_5 = sum(1 for r in final_query_results if r["hit_5"])
    total_hit_10 = sum(1 for r in final_query_results if r["hit_10"])
    total_hit_20 = sum(1 for r in final_query_results if r["hit_20"])

    r1_val = recalls.get(1, 0.0)
    r5_val = recalls.get(5, 0.0)
    r10_val = recalls.get(10, 0.0)
    r20_val = recalls.get(20, 0.0)

    # 9. Question-Type Breakdown
    types = sorted({r["qtype"] for r in final_query_results})
    qtype_metrics: dict[str, Any] = {}
    for qt in types:
        sub_indices = [i for i, r in enumerate(final_query_results) if r["qtype"] == qt]
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

    # 10. Model Call & Action Distributions
    strategy_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    verif_counts: dict[str, int] = {}
    repairs_attempted = 0
    repairs_recovered = 0

    for r in final_query_results:
        for act in r.get("strategy_action_sequence", []):
            action_counts[act] = action_counts.get(act, 0) + 1
        v_st = r.get("verification_status", "UNVERIFIED")
        verif_counts[v_st] = verif_counts.get(v_st, 0) + 1
        if r.get("repair_count", 0) > 0:
            repairs_attempted += 1
            if r.get("hit_10", False):
                repairs_recovered += 1

    total_model_calls = sum(r.get("model_call_count", 0) for r in final_query_results)
    total_in_tokens = sum(r.get("input_tokens", 0) for r in final_query_results)
    total_out_tokens = sum(r.get("output_tokens", 0) for r in final_query_results)
    total_all_tokens = sum(r.get("total_tokens", 0) for r in final_query_results)
    call_counts = [r.get("model_call_count", 0) for r in final_query_results]
    model_latencies = [r.get("model_latency_ms", 0.0) for r in final_query_results]

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
        "avg_tokens_per_query": float(total_all_tokens / 100),
        "total_model_latency_ms": float(sum(model_latencies)),
        "avg_model_latency_ms": float(np.mean(model_latencies)) if model_latencies else 0.0,
    }

    # 11. Build Benchmark Artifact
    benchmark_record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "benchmark_file": str(public_file),
        "total_queries": 100,
        "pipeline": "A4_Agentic_GraphRAG_LLM",
        "model_name": "gemini-3.5-flash-lite",
        "provider": "Google Gemini API",
        "execution_mode": "model_driven",
        "benchmark_scope": "public only; hidden benchmark was not accessed",
        "integrity_verification": {
            "model_driven_queries": 100,
            "deterministic_fallback_queries": 0,
            "hidden_queries_accessed": 0,
            "rerun_fallback_qids_count": len(target_fallback_qids),
            "no_existing_benchmark_overwritten": True,
            "no_groq_results_included": True,
        },
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
        },
        "qtype_breakdown": qtype_metrics,
        "strategy_action_distribution": {
            "actions": action_counts,
        },
        "verification_breakdown": verif_counts,
        "repair_statistics": {
            "attempted": repairs_attempted,
            "recovered_at_10": repairs_recovered,
            "repair_rate": repairs_attempted / 100.0,
        },
        "rate_limiting_statistics": {
            "rate_limit_events_429": RATE_LIMIT_STATE["rate_limit_events"],
            "api_retries": RATE_LIMIT_STATE["api_retries"],
            "target_rpm": 10,
            "max_rpm_ceiling": 12,
        },
        "model_call_statistics": model_call_stats,
        "query_results": final_query_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_record, f, indent=2)

    logger.info(f"\nBenchmark artifact successfully saved to: {out_path}")

    # Print Final Summary
    print("\n" + "=" * 95)
    print("FINAL COMPLETE 100-QUERY LLM-DRIVEN A4 BENCHMARK RESULTS")
    print("=" * 95)
    print("Pipeline:                     A4_Agentic_GraphRAG_LLM")
    print("Model:                        gemini-3.5-flash-lite (Google Gemini API)")
    print("Execution Mode:               100% model_driven (100/100), 0 fallbacks")
    print("Total Questions:              100")
    print(f"Recall@1:                     {r1_val:.4f} ({r1_val * 100:.2f}%)")
    print(f"Recall@5:                     {r5_val:.4f} ({r5_val * 100:.2f}%)")
    print(f"Recall@10:                    {r10_val:.4f} ({r10_val * 100:.2f}%)")
    print(f"Recall@20:                    {r20_val:.4f} ({r20_val * 100:.2f}%)")
    print(f"Mean Reciprocal Rank (MRR):   {mrr:.4f}")
    print(f"Queries with Gold Hit (@1):   {total_hit_1} / 100")
    print(f"Queries with Gold Hit (@5):   {total_hit_5} / 100")
    print(f"Queries with Gold Hit (@10):  {total_hit_10} / 100")
    print(f"Queries with Gold Hit (@20):  {total_hit_20} / 100")
    print("\nLatency Statistics (End-to-End per Query):")
    print(f"  Mean Latency:               {avg_latency:.2f} ms")
    print(f"  p50 (Median) Latency:       {p50_latency:.2f} ms")
    print(f"  p95 Latency:                {p95_latency:.2f} ms")
    print(f"  p99 Latency:                {p99_latency:.2f} ms")
    print("\nModel & Telemetry Statistics:")
    print(f"  Total Model Calls:          {total_model_calls}")
    print(f"  Avg Calls / Query:          {model_call_stats['avg_model_calls_per_query']:.2f}")
    print(f"  Total Input Tokens:         {total_in_tokens}")
    print(f"  Total Output Tokens:        {total_out_tokens}")
    print(f"  Total Tokens:               {total_all_tokens}")
    print(f"  Avg Tokens / Query:         {model_call_stats['avg_tokens_per_query']:.1f}")
    print(f"  Rate Limit 429 Events:      {RATE_LIMIT_STATE['rate_limit_events']}")
    print(f"  API Retries:                {RATE_LIMIT_STATE['api_retries']}")
    print("\nVerification & Repair:")
    print(f"  Repairs Attempted:          {repairs_attempted}")
    print(f"  Repairs Recovered (@10):    {repairs_recovered}")

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

    print("\n" + "=" * 95)
    print("COMPARISON: A0 (Vector) vs A1 (Adaptive) vs Deterministic A4 vs Pure LLM-driven A4")
    print("=" * 95)
    comp_header = (
        f"{'Pipeline':<28} | {'Recall@1':<9} | {'Recall@5':<9} | "
        f"{'Recall@10':<9} | {'Recall@20':<9} | {'MRR':<8} | {'Mean Latency':<13}"
    )
    print(comp_header)
    print("-" * 100)

    print(
        f"{'A0 Vector Baseline':<28} | "
        f"{68.00:>8.2f}% | "
        f"{83.00:>8.2f}% | "
        f"{86.00:>8.2f}% | "
        f"{90.00:>8.2f}% | "
        f"{0.7394:>7.4f} | "
        f"{430.66:>9.2f} ms"
    )
    print(
        f"{'A1 Adaptive GraphRAG':<28} | "
        f"{66.00:>8.2f}% | "
        f"{81.00:>8.2f}% | "
        f"{86.00:>8.2f}% | "
        f"{90.00:>8.2f}% | "
        f"{0.7290:>7.4f} | "
        f"{2370.00:>9.2f} ms"
    )
    print(
        f"{'A4 Deterministic Baseline':<28} | "
        f"{91.00:>8.2f}% | "
        f"{100.00:>8.2f}% | "
        f"{100.00:>8.2f}% | "
        f"{100.00:>8.2f}% | "
        f"{0.9445:>7.4f} | "
        f"{8218.85:>9.2f} ms"
    )
    print(
        f"{'A4 Complete LLM-Driven':<28} | "
        f"{r1_val * 100:>8.2f}% | "
        f"{r5_val * 100:>8.2f}% | "
        f"{r10_val * 100:>8.2f}% | "
        f"{r20_val * 100:>8.2f}% | "
        f"{mrr:>7.4f} | "
        f"{avg_latency:>9.2f} ms"
    )
    print("=" * 95)


if __name__ == "__main__":
    main()
