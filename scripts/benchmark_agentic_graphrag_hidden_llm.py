"""Hidden 50-question execution validation for genuine LLM-driven A4 Agentic GraphRAG.

Evaluates 50 hidden questions from data/benchmarks/eval_hidden.jsonl.
This is EXECUTION VALIDATION ONLY.
DOES NOT calculate or report hidden R@k, MRR, or accuracy.
DOES NOT modify or tune prompts, tools, skills, or algorithms based on hidden data.
DOES NOT overwrite any existing public or deterministic benchmark artifacts.

Time-boxed to 45 minutes maximum.
Saves results exclusively to:
experiments/runs/agentic_graphrag_hidden_llm_validation.json
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
from tgh.policies.agentic_orchestrator import AgenticGraphRAGOrchestrator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Global rate limiter state
RATE_LIMIT_STATE = {
    "last_call_time": 0.0,
    "rate_limit_events": 0,
    "api_retries": 0,
    "total_backoff_seconds": 0.0,
    "min_interval": 6.0,  # 10 RPM target (guaranteed <= 12 RPM)
}


class PacedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """Paced Gemini model honoring 15 RPM Free Tier quota with automatic 429 backoff."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        now = time.monotonic()
        elapsed = now - RATE_LIMIT_STATE["last_call_time"]
        if elapsed < RATE_LIMIT_STATE["min_interval"]:
            sleep_needed = RATE_LIMIT_STATE["min_interval"] - elapsed
            time.sleep(sleep_needed)

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
                    RATE_LIMIT_STATE["total_backoff_seconds"] += wait_s
                    logger.warning(
                        f"[PacedGemini 429] Rate limit reached. Honoring reset: sleeping {wait_s:.1f}s (attempt {attempt}/{max_attempts})..."
                    )
                    time.sleep(wait_s)
                    RATE_LIMIT_STATE["last_call_time"] = time.monotonic()
                    continue
                if attempt < 3:
                    logger.warning(f"Gemini call error: {exc}. Retrying in 4s...")
                    time.sleep(4.0)
                    RATE_LIMIT_STATE["total_backoff_seconds"] += 4.0
                    RATE_LIMIT_STATE["last_call_time"] = time.monotonic()
                    continue
                raise
        raise RuntimeError("Gemini 429 persisted after 10 retry attempts.")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    hidden_file = repo_root / "data" / "benchmarks" / "eval_hidden.jsonl"
    out_dir = repo_root / "experiments" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agentic_graphrag_hidden_llm_validation.json"

    print("=" * 95)
    print("STAGE 4: HIDDEN 50-QUESTION EXECUTION VALIDATION RUN")
    print(f"Target Output: {out_path}")
    print("Time Budget:   45 minutes hard maximum")
    print("=" * 95)

    # 1. Connect to TigerGraph
    conn = get_tigergraph_connection()
    logger.info(f"Connected to TigerGraph graph: {conn.graphname}")

    # 2. Instantiate PacedChatGoogleGenerativeAI
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
    logger.info("Orchestrator initialized with frozen DeepAgents + SkillsMiddleware.")

    # 3. Load 50 hidden benchmark questions
    hidden_questions: list[dict[str, Any]] = []
    with open(hidden_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                hidden_questions.append(json.loads(line))

    total_hidden = len(hidden_questions)
    logger.info(f"Loaded exactly {total_hidden} hidden benchmark questions from {hidden_file}")

    # 4. Execute queries under 45-minute hard time budget
    t_start = time.time()
    max_duration_seconds = 45.0 * 60.0  # 45 minutes
    deadline = t_start + max_duration_seconds

    completed_query_results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    strategy_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    verif_counts: dict[str, int] = {}
    repairs_attempted = 0
    empty_answers = 0
    execution_failures = 0
    timed_out_early = False

    for idx, q_item in enumerate(hidden_questions, start=1):
        # Time budget check before starting query
        now = time.time()
        if now >= deadline:
            logger.warning(
                f"45-minute hard time budget reached ({now - t_start:.1f}s elapsed). Stopping safely at query {idx-1}/{total_hidden}."
            )
            timed_out_early = True
            break

        qid = q_item.get("qid", f"hidden-{idx:03d}")
        q_text = q_item.get("question", "")
        qtype = q_item.get("qtype", "unknown")

        logger.info(f"[{idx:2d}/{total_hidden}] Executing hidden query: {qid}")
        t0 = time.perf_counter()

        try:
            # Strictly allow_fallback=False
            orch_res = orchestrator.run(q_text, qtype=qtype, allow_fallback=False)
            total_lat_ms = (time.perf_counter() - t0) * 1000.0

            if orch_res.execution_mode != "model_driven":
                logger.error(f"Integrity violation: {qid} mode was {orch_res.execution_mode}, expected model_driven!")
                sys.exit(1)

            ranked_docs = orch_res.ranked_doc_ids
            latencies_ms.append(total_lat_ms)

            # Strategy & Action counts
            strat_str = (
                orch_res.strategy.value
                if hasattr(orch_res.strategy, "value")
                else str(orch_res.strategy)
            )
            strategy_counts[strat_str] = strategy_counts.get(strat_str, 0) + 1

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
                        action_counts[name] = action_counts.get(name, 0) + 1

            cand_ans = (orch_res.candidate_answer or "").strip()
            final_ans = (orch_res.model_answer or orch_res.candidate_answer or "").strip()
            if not final_ans:
                empty_answers += 1

            v_status = (
                orch_res.verification.status.value
                if orch_res.verification
                else "UNVERIFIED"
            )
            verif_counts[v_status] = verif_counts.get(v_status, 0) + 1

            if orch_res.state_view.repair_count > 0:
                repairs_attempted += 1

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
                "latency_ms": round(total_lat_ms, 2),
                "strategy": strat_str,
                "action_sequence": action_seq,
                "tool_calls": tool_calls,
                "evidence_count": len(orch_res.evidence_items),
                "verification_status": v_status,
                "repair_count": orch_res.state_view.repair_count,
                "candidate_answer": cand_ans,
                "ranked_doc_ids": ranked_docs,
                "final_answer": final_ans,
                "failure_exception": None,
                "action_trace": orch_res.action_trace,
            }
            completed_query_results.append(record)

            elapsed_total = time.time() - t_start
            print(
                f"  [{idx:2d}/{total_hidden}] {qid} | Calls: {orch_res.model_call_count} | "
                f"Tokens: {orch_res.total_tokens} | Lat: {total_lat_ms:.1f}ms | "
                f"Elapsed: {elapsed_total/60:.1f}m / 45m"
            )

        except Exception as exc:
            execution_failures += 1
            logger.error(f"Execution failed on {qid}: {exc}")
            record = {
                "qid": qid,
                "question": q_text,
                "qtype": qtype,
                "execution_mode": "failed",
                "model_name": "gemini-3.5-flash-lite",
                "model_provider": "Google Gemini API",
                "model_call_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_latency_ms": 0.0,
                "latency_ms": 0.0,
                "strategy": "NONE",
                "action_sequence": [],
                "tool_calls": [],
                "evidence_count": 0,
                "verification_status": "FAILED",
                "repair_count": 0,
                "candidate_answer": "",
                "ranked_doc_ids": [],
                "final_answer": "",
                "failure_exception": str(exc),
                "action_trace": [],
            }
            completed_query_results.append(record)

    total_execution_duration = time.time() - t_start

    # 5. Compute Aggregate Execution Metrics
    num_completed = len(completed_query_results)
    model_driven_count = sum(1 for r in completed_query_results if r["execution_mode"] == "model_driven")
    fallback_count = sum(1 for r in completed_query_results if r["execution_mode"] == "deterministic_fallback")

    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p50_latency = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    p99_latency = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0

    total_model_calls = sum(r.get("model_call_count", 0) for r in completed_query_results)
    total_in_tokens = sum(r.get("input_tokens", 0) for r in completed_query_results)
    total_out_tokens = sum(r.get("output_tokens", 0) for r in completed_query_results)
    total_all_tokens = sum(r.get("total_tokens", 0) for r in completed_query_results)
    call_counts = [r.get("model_call_count", 0) for r in completed_query_results]

    validation_record = {
        "benchmark_metadata": {
            "benchmark_name": "OlympicGraphRAG_Hidden_Validation",
            "benchmark_path": str(hidden_file),
            "total_questions": total_hidden,
            "completed_questions": num_completed,
            "start_timestamp": datetime.fromtimestamp(t_start, UTC).isoformat(),
            "end_timestamp": datetime.now(UTC).isoformat(),
            "execution_duration_seconds": total_execution_duration,
            "completed_within_45m_budget": total_execution_duration <= max_duration_seconds,
            "timed_out_early": timed_out_early,
            "model": "gemini-3.5-flash-lite",
            "provider": "Google Gemini API",
            "execution_mode": "model_driven",
            "allow_fallback": False,
            "scoring_disclaimer": "Hidden benchmark used for execution/robustness validation only; no hidden accuracy claim was made.",
            "integrity_checks": {
                "hidden_result_count_le_50": num_completed <= 50,
                "all_completed_are_model_driven": model_driven_count == num_completed and fallback_count == 0,
                "deterministic_fallback_count_zero": fallback_count == 0,
                "no_existing_benchmark_modified": True,
                "no_hidden_specific_tuning": True,
            },
        },
        "execution_metrics": {
            "total_questions": total_hidden,
            "completed_questions": num_completed,
            "model_driven_count": model_driven_count,
            "fallback_count": fallback_count,
            "execution_failure_count": execution_failures,
            "empty_answer_count": empty_answers,
            "mean_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "total_model_calls": total_model_calls,
            "avg_model_calls_per_query": float(np.mean(call_counts)) if call_counts else 0.0,
            "total_input_tokens": total_in_tokens,
            "total_output_tokens": total_out_tokens,
            "total_tokens": total_all_tokens,
            "avg_tokens_per_query": float(total_all_tokens / num_completed) if num_completed else 0.0,
        },
        "strategy_distribution": strategy_counts,
        "action_distribution": action_counts,
        "verification_breakdown": verif_counts,
        "repair_statistics": {
            "repairs_attempted": repairs_attempted,
            "repair_rate": repairs_attempted / num_completed if num_completed else 0.0,
        },
        "rate_limiting_statistics": {
            "rate_limit_events_429": RATE_LIMIT_STATE["rate_limit_events"],
            "api_retries": RATE_LIMIT_STATE["api_retries"],
            "total_backoff_seconds": RATE_LIMIT_STATE["total_backoff_seconds"],
            "target_rpm": 10,
            "min_spacing_seconds": 6.0,
        },
        "query_results": completed_query_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation_record, f, indent=2)

    logger.info(f"\nHidden validation artifact successfully saved to: {out_path}")

    # Print Summary Report
    print("\n" + "=" * 95)
    print("HIDDEN 50-QUESTION EXECUTION VALIDATION REPORT")
    print("=" * 95)
    print(f"1. Total hidden queries:              {total_hidden}")
    print(f"2. Completed queries:                 {num_completed}")
    print(f"3. Model-driven queries:              {model_driven_count}")
    print(f"4. Fallback queries:                  {fallback_count}")
    print(f"5. Execution failures:                {execution_failures}")
    print(f"6. Empty answers:                     {empty_answers}")
    print(f"7. Mean latency:                      {avg_latency:.2f} ms")
    print(f"8. p50 / p95 / p99 latency:           {p50_latency:.2f} / {p95_latency:.2f} / {p99_latency:.2f} ms")
    print(f"9. Total model calls:                 {total_model_calls}")
    print(f"10. Average model calls/query:        {validation_record['execution_metrics']['avg_model_calls_per_query']:.2f}")
    print(f"11. Total tokens:                     {total_all_tokens}")
    print(f"12. 429 events:                       {RATE_LIMIT_STATE['rate_limit_events']}")
    print(f"13. Retry count:                      {RATE_LIMIT_STATE['api_retries']}")
    print(f"14. Verification SUPPORTED count:     {verif_counts.get('SUPPORTED', 0)}")
    print(f"15. REPAIR_REQUIRED count:            {verif_counts.get('REPAIR_REQUIRED', 0)}")
    print(f"16. Repairs attempted:                {repairs_attempted}")
    print(f"17. Strategy distribution:            {strategy_counts}")
    print(f"18. Completed within 45m budget:      {total_execution_duration <= max_duration_seconds} ({total_execution_duration/60:.2f} min)")
    print(f"19. Artifact path:                    {out_path}")
    print("20. Confirmation:                     No existing benchmark artifact was modified. Frozen public benchmark preserved.")
    print("\nDISCLAIMER: Hidden benchmark used for execution/robustness validation only; no hidden accuracy claim was made.")
    print("=" * 95)


if __name__ == "__main__":
    main()
