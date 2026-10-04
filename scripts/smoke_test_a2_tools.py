"""Live smoke test script for Stage 3 / A2 deterministic graph tools.

Tests targeted public benchmark multi-hop cases on the live OlympicGraphRAG instance.
Saves results to experiments/runs/a2_graph_tools_smoke.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from scripts.ingest_graph_entities import get_tigergraph_connection
from tgh.mcp.graph_tools import TigerGraphTools
from tgh.telemetry.trace import TraceRecorder


def run_live_smoke():
    print("=" * 70)
    print("STAGE 3 / A2 DETERMINISTIC GRAPH TOOLS LIVE SMOKE TEST")
    print("=" * 70)

    conn = get_tigergraph_connection()
    recorder = TraceRecorder(run_id="a2-live-smoke")
    tools = TigerGraphTools(conn=conn, recorder=recorder)

    # Targeted multi-hop public questions from eval_public.jsonl
    test_cases: list[dict[str, Any]] = [
        # Test 1: Venue -> Event -> Participant (Riocentro Pavilion 4 -> Carolina Marin)
        {
            "qid": "pub-014",
            "question": (
                "Who won the gold medal in the event held at Riocentro – Pavilion 4 "
                "on 11–19 August at the 2016 Summer Olympics?"
            ),
            "operation": "execute_composite_multihop",
            "inputs": {
                "venue_id": "venue_riocentro_pavilion_4",
                "year": 2016,
                "sport": "badminton",
                "event_name_query": "women's singles",
            },
            "gold_doc_ids": ["Q25301483"],
            "expected_answer": "Carolina Marín",
        },
        # Test 2: Participant -> Event Filtering (Disambiguate Michael Phelps 2004)
        {
            "qid": "pub-028",
            "question": (
                "Who won the gold medal in the event held at Olympic Aquatic Centre "
                "on August 14, 2004 (heats & final)?"
            ),
            "operation": "execute_composite_multihop",
            "inputs": {
                "participant_type": "Person",
                "participant_id": "person_michael_phelps",
                "year": 2004,
                "event_name_query": "400 metre individual medley",
            },
            "gold_doc_ids": ["Q1141105"],
            "expected_answer": "Michael Phelps",
        },
        # Test 3: Venue -> Event -> Participant (London Velopark 2012 -> Cycling)
        {
            "qid": "pub-015",
            "question": (
                "Who won the gold medal in the event held at London Velopark "
                "on 3 to 4 August at the 2012 Summer Olympics?"
            ),
            "operation": "execute_composite_multihop",
            "inputs": {
                "venue_id": "venue_london_velopark",
                "year": 2012,
                "sport": "cycling",
                "event_name_query": "team pursuit",
            },
            "gold_doc_ids": ["Q2297633"],
            "expected_answer": "Dani KingLaura TrottJoanna Rowsell",
        },
        # Test 4: Venue -> Event -> Participant (Richmond Olympic Oval 2010 -> Skating)
        {
            "qid": "pub-011",
            "question": (
                "Who won the gold medal in the event held at Richmond Olympic Oval "
                "on 14 February 2010?"
            ),
            "operation": "execute_composite_multihop",
            "inputs": {
                "venue_id": "venue_richmond_olympic_oval",
                "year": 2010,
                "sport": "speed skating",
                "event_name_query": "3000 metres",
            },
            "gold_doc_ids": ["Q580481"],
            "expected_answer": "Martina Sáblíková",
        },
        # Test 5: Venue -> Event -> Participant (Royal Artillery Barracks 2012)
        {
            "qid": "pub-017",
            "question": (
                "Who won the gold medal in the event held at Royal Artillery Barracks "
                "on 28 July 2012?"
            ),
            "operation": "execute_composite_multihop",
            "inputs": {
                "venue_id": "venue_royal_artillery_barracks",
                "year": 2012,
                "sport": "shooting",
                "event_name_query": "10 metre air rifle",
            },
            "gold_doc_ids": ["Q1137721"],
            "expected_answer": "Yi Siling",
        },
        # Test 6: Country/Participant Aggregation (Weightlifting Gymnasium 1988)
        {
            "qid": "pub-005",
            "question": (
                "Who won the gold medal in the event held at Olympic Weightlifting "
                "Gymnasium on 20 September 1988?"
            ),
            "operation": "execute_composite_multihop",
            "inputs": {
                "venue_id": "venue_olympic_weightlifting_gymnasium",
                "year": 1988,
                "event_name_query": "60 kg",
            },
            "gold_doc_ids": ["Q25239316"],
            "expected_answer": "Naim Süleymanoğlu",
        },
    ]

    results_out: list[dict[str, Any]] = []

    for tc in test_cases:
        qid = tc["qid"]
        op = tc["operation"]
        inputs = tc["inputs"]
        print(f"\nEvaluating {qid}: {tc['question'][:65]}...")

        t0 = time.perf_counter()
        res = tools.execute_composite_multihop(
            participant_type=inputs.get("participant_type"),
            participant_id=inputs.get("participant_id"),
            venue_id=inputs.get("venue_id"),
            year=inputs.get("year"),
            sport=inputs.get("sport"),
            event_name_query=inputs.get("event_name_query"),
            include_participants=True,
            include_countries=True,
        )
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        events = res.data.get("events", []) if res.success and res.data else []
        contexts = (
            res.data.get("event_contexts", []) if res.success and res.data else []
        )
        country_agg = (
            res.data.get("country_aggregation") if res.success and res.data else None
        )

        surviving_eids = [e.event_id for e in events]
        gold_reached = any(g in surviving_eids for g in tc["gold_doc_ids"])

        # Check participant names found
        found_participants: list[str] = []
        for ctx in contexts:
            for p in ctx.participants:
                found_participants.append(p["name"])

        item = {
            "qid": qid,
            "question": tc["question"],
            "operation": op,
            "inputs": inputs,
            "gold_doc_ids": tc["gold_doc_ids"],
            "expected_answer": tc["expected_answer"],
            "success": res.success,
            "gold_reached": gold_reached,
            "candidate_event_ids": surviving_eids,
            "candidate_event_names": [e.name for e in events],
            "candidate_participants": found_participants,
            "country_aggregation": (country_agg.to_dict() if country_agg else None),
            "provenance_paths": res.provenance[:10],
            "latency_ms": t_elapsed_ms,
            "failure_reason": res.error
            if not res.success
            else (
                None
                if gold_reached
                else f"Gold {tc['gold_doc_ids']} not in {surviving_eids}"
            ),
        }
        results_out.append(item)

        status_str = "REACHED" if gold_reached else "MISSED"
        print(
            f"  Result: [{status_str}] | Latency: {t_elapsed_ms:.1f}ms | "
            f"Events: {len(surviving_eids)} | Participants: {found_participants[:3]}"
        )

    # Save to experiments/runs/a2_graph_tools_smoke.json
    out_path = Path("experiments/runs/a2_graph_tools_smoke.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_tested": len(test_cases),
                "gold_reached_count": sum(1 for r in results_out if r["gold_reached"]),
                "average_latency_ms": (
                    sum(r["latency_ms"] for r in results_out) / len(results_out)
                ),
                "results": results_out,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("\n" + "=" * 70)
    print(f"Artifact saved to: {out_path.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    run_live_smoke()
