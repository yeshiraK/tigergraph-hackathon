"""Smoke test for Stage 4 A3 Verification + A4 Agentic GraphRAG Orchestration.

Executes 6 representative public benchmark questions against live TigerGraph Cloud
with strict machine-readable telemetry and saves to experiments/runs/a3_a4_smoke.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Add root and src to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from scripts.ingest_graph_entities import get_tigergraph_connection  # noqa: E402
from tgh.policies.agentic_orchestrator import (  # noqa: E402
    AgenticGraphRAGOrchestrator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SMOKE_QUESTIONS = [
    {
        "qid": "pub-009",
        "category": "simple_lookup",
        "question": (
            "How many nations competed in Sailing at the "
            "2016 Summer Olympics – Women's RS:X?"
        ),
        "gold_doc_ids": ["Q26254891"],
        "expected_answer": ["26"],
    },
    {
        "qid": "pub-002",
        "category": "temporal",
        "question": (
            "Who won the gold medal in the men's 20 kilometres walk athletics event "
            "at the Summer Olympics held immediately before 2016?"
        ),
        "gold_doc_ids": ["Q1050909", "Q26233122"],
        "expected_answer": ["Chen Ding"],
    },
    {
        "qid": "pub-007",
        "category": "relational_temporal",
        "question": (
            "Who won the gold medal in the men's pole vault athletics event at "
            "the Summer Olympics held immediately before 2016?"
        ),
        "gold_doc_ids": ["Q2000968", "Q26208457"],
        "expected_answer": ["Renaud Lavillenie"],
    },
    {
        "qid": "pub-014",
        "category": "venue_event_multihop",
        "question": (
            "Who won the gold medal in the event held at Riocentro – Pavilion 4 "
            "on 11–19 August at the 2016 Summer Olympics?"
        ),
        "gold_doc_ids": ["Q25301483"],
        "expected_answer": ["Carolina Marín"],
    },
    {
        "qid": "pub-028",
        "category": "participant_multievent_disambiguation",
        "question": (
            "Who won the gold medal in the event held at Olympic Aquatic Centre "
            "on August 14, 2004 (heats & final)?"
        ),
        "gold_doc_ids": ["Q1141105"],
        "expected_answer": ["Michael Phelps"],
    },
    {
        "qid": "pub-032",
        "category": "aggregation_country",
        "question": (
            "How many nations competed in Gymnastics at the "
            "2016 Summer Olympics – Men's horizontal bar?"
        ),
        "gold_doc_ids": ["Q26233795"],
        "expected_answer": ["34"],
    },
]


def main() -> None:
    logger.info("Initializing live TigerGraph connection for A3/A4 smoke test...")
    conn = get_tigergraph_connection()
    orchestrator = AgenticGraphRAGOrchestrator(conn=conn)

    results = []
    logger.info(f"Executing {len(SMOKE_QUESTIONS)} smoke test questions...")

    for item in SMOKE_QUESTIONS:
        qid = item["qid"]
        category = item["category"]
        question = item["question"]
        gold_doc_ids = item["gold_doc_ids"]

        logger.info(f"\n--- Running [{qid}] ({category}) ---")
        logger.info(f"Question: {question}")

        t0 = time.perf_counter()
        orch_res = orchestrator.run(question)
        elapsed = time.perf_counter() - t0

        evidence_ids = [it.evidence_id for it in orch_res.evidence_items]
        provenances = [
            it.provenance_path for it in orch_res.evidence_items if it.provenance_path
        ]

        # Calculate hits against gold_doc_ids
        r1_hit = bool(
            orch_res.ranked_doc_ids and orch_res.ranked_doc_ids[0] in gold_doc_ids
        )
        r5_hit = any(d in gold_doc_ids for d in orch_res.ranked_doc_ids[:5])
        r10_hit = any(d in gold_doc_ids for d in orch_res.ranked_doc_ids[:10])

        record = {
            "qid": qid,
            "category": category,
            "question": question,
            "selected_strategy": orch_res.strategy.value,
            "skills_invoked": orch_res.skills_invoked,
            "mcp_tools_invoked": orch_res.mcp_tools_invoked,
            "a2_tools_invoked": orch_res.a2_tools_invoked,
            "verification_result": (
                orch_res.verification.status.value if orch_res.verification else "NONE"
            ),
            "verification_explanation": (
                orch_res.verification.explanation if orch_res.verification else ""
            ),
            "repair_used": "yes" if orch_res.repair_used else "no",
            "candidate_answer": orch_res.candidate_answer,
            "ranked_doc_ids": orch_res.ranked_doc_ids[:10],
            "gold_doc_ids": gold_doc_ids,
            "r1_hit": r1_hit,
            "r5_hit": r5_hit,
            "r10_hit": r10_hit,
            "evidence_count": len(orch_res.evidence_items),
            "evidence_ids": evidence_ids[:10],
            "provenance": provenances[:5],
            "latency_ms": round(orch_res.latency_ms, 2),
            "total_elapsed_s": round(elapsed, 3),
            "run_id": orch_res.run_id,
        }
        results.append(record)
        logger.info(
            f"Result: strategy={record['selected_strategy']}, "
            f"skills={record['skills_invoked']}, "
            f"verif={record['verification_result']}, repair={record['repair_used']}, "
            f"R@1={r1_hit}, R@5={r5_hit}, latency={record['latency_ms']}ms"
        )

    out_dir = Path("experiments/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "a3_a4_smoke.json"

    n_q = len(results)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_questions": n_q,
                "summary": {
                    "r1_accuracy": sum(1 for r in results if r["r1_hit"]) / n_q,
                    "r5_accuracy": sum(1 for r in results if r["r5_hit"]) / n_q,
                    "r10_accuracy": sum(1 for r in results if r["r10_hit"]) / n_q,
                    "repair_used_count": sum(
                        1 for r in results if r["repair_used"] == "yes"
                    ),
                    "avg_latency_ms": sum(r["latency_ms"] for r in results) / n_q,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"\nSmoke test complete! Results saved to {out_file}")


if __name__ == "__main__":
    main()
