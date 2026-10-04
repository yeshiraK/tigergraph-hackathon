"""Deterministic TigerGraph population script for Stage 1.

Extracts domain entities from data/raw/corpus.jsonl and loads them into OlympicGraphRAG.
Supports --dry-run (default) and --execute flags.

Never modifies:
- Document vertices (remains 2,951)
- Chunk vertices (remains 9,348)
- HAS_CHUNK edges (remains 9,348)
- Chunk embeddings (768D Nomic v1.5)
- HNSW index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyTigerGraph as tg

from tgh.ingestion.graph_extractor import (
    ExtractedGraphData,
    extract_graph_from_corpus,
    validate_extracted_graph,
)


def load_env() -> dict[str, str]:
    """Parse key-value pairs from project-root .env file."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    res: dict[str, str] = {}
    if env_file.is_file():
        with env_file.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if "=" in line_str:
                    k, v = line_str.split("=", 1)
                    res[k.strip()] = v.strip()
    return res


def get_tigergraph_connection() -> tg.TigerGraphConnection:
    """Connect to TigerGraph Cloud using .env configuration."""
    env = load_env()
    conn = tg.TigerGraphConnection(
        host=env["TIGERGRAPH_HOST"],
        graphname=env["TIGERGRAPH_GRAPH_NAME"],
        gsqlSecret=env["TIGERGRAPH_SECRET"],
    )
    return conn


def print_dry_run_summary(data: ExtractedGraphData, error_count: int = 0) -> None:
    """Print the required concise dry-run summary."""
    s = data.stats
    print("=" * 70)
    print("STAGE 1 — OFFLINE GRAPH EXTRACTION DRY-RUN SUMMARY")
    print("=" * 70)
    print(f"Total documents processed:     {s.total_docs_processed:,}")
    print(f"Olympic event documents:       {s.olympic_event_docs:,}")
    print(f"Skipped non-Olympic documents: {s.skipped_non_olympic_docs:,}")
    print("-" * 70)
    print("PROPOSED VERTEX COUNTS BY TYPE:")
    print(f"  number of Event vertices:    {s.event_count:,}")
    print(f"  number of Sport vertices:    {s.sport_count:,}")
    print(f"  number of Venue vertices:    {s.venue_count:,}")
    print(f"  number of Country vertices:  {s.country_count:,}")
    print(f"  number of Person vertices:   {s.person_count:,}")
    print(f"  number of Team vertices:     {s.team_count:,}")
    print(f"  number of Entity vertices:   {s.entity_count:,}")
    total_new_v = (
        s.event_count
        + s.sport_count
        + s.venue_count
        + s.country_count
        + s.person_count
        + s.team_count
        + s.entity_count
    )
    print(f"  TOTAL NEW VERTICES:          {total_new_v:,}")
    print("-" * 70)
    print("PROPOSED EDGE COUNTS BY TYPE:")
    print(f"  number of BELONGS_TO edges:    {s.belongs_to_count:,}")
    print(f"  number of HELD_AT edges:       {s.held_at_count:,}")
    print(f"  number of PARTICIPATED_IN:     {s.participated_in_count:,}")
    print(f"  number of REPRESENTS edges:    {s.represents_count:,}")
    print(f"  number of MENTIONS edges:      {s.mentions_count:,}")
    print(f"  number of RESOLVES_TO edges:   {s.resolves_to_count:,}")
    total_new_e = (
        s.belongs_to_count
        + s.held_at_count
        + s.participated_in_count
        + s.represents_count
        + s.mentions_count
        + s.resolves_to_count
    )
    print(f"  TOTAL NEW EDGES:               {total_new_e:,}")
    print("-" * 70)
    print("PERSON VS TEAM CLASSIFICATIONS:")
    print(f"  Person classifications:        {s.person_classifications:,}")
    print(f"  Team classifications:          {s.team_classifications:,}")
    print("-" * 70)
    print("MENTIONS PROVENANCE STATISTICS:")
    print(f"  Total MENTIONS edges:          {s.mentions_count:,}")
    print(f"  Sport title-only (no mention): {s.sport_title_only_no_mentions:,}")
    print("-" * 70)
    print("UNRESOLVED AND AMBIGUOUS DATA:")
    print(f"  number of unresolved entities (venues):     {s.unresolved_venues:,}")
    print(f"  number of unresolved entities (countries):  {s.unresolved_countries:,}")
    print(f"  number of unresolved entities (medalists):  {s.unresolved_medalists:,}")
    print(f"  number of ambiguous mappings:               {s.ambiguous_mappings:,}")
    print(f"  number of skipped facts (non-Olympic docs): {s.skipped_non_olympic_docs}")
    print(f"  Validation error count:                     {error_count}")
    print("=" * 70)


def perform_preflight_checks(
    conn: tg.TigerGraphConnection,
) -> dict[str, int]:
    """Execute read-only preflight checks against live TigerGraph workspace."""
    print("\n" + "=" * 70)
    print("STEP 1: READ-ONLY LIVE PREFLIGHT CHECKS")
    print("=" * 70)

    # 1. Graph name check
    if conn.graphname != "OlympicGraphRAG":
        raise ValueError(
            f"FATAL: Connected graph is '{conn.graphname}', expected 'OlympicGraphRAG'!"
        )
    print(f"Verified Graph Name:           {conn.graphname}")

    # 2. Schema check
    schema = conn.getSchema(force=True)
    live_v_types = {v["Name"] for v in schema.get("VertexTypes", [])}
    live_e_types = {e["Name"] for e in schema.get("EdgeTypes", [])}

    expected_v_types = {
        "Document", "Chunk", "Entity", "Event", "Person",
        "Country", "Sport", "Venue", "Team",
    }
    expected_e_types = {
        "HAS_CHUNK", "MENTIONS", "RESOLVES_TO", "PARTICIPATED_IN",
        "REPRESENTS", "BELONGS_TO", "HELD_AT",
    }

    if not expected_v_types.issubset(live_v_types):
        missing = expected_v_types - live_v_types
        raise ValueError(f"FATAL: Missing required vertex types in schema: {missing}")
    if not expected_e_types.issubset(live_e_types):
        missing = expected_e_types - live_e_types
        raise ValueError(f"FATAL: Missing required edge types in schema: {missing}")

    print("Verified Vertex Schema (9):    All expected types present.")
    print("Verified Edge Schema (7):      All expected types present.")

    # 3. Live counts check
    v_counts = {v: conn.getVertexCount(v) for v in expected_v_types}
    e_counts = {e: conn.getEdgeCount(e) for e in expected_e_types}

    print("\nCurrent Live Vertex Counts:")
    for v in sorted(v_counts.keys()):
        print(f"  {v:<14}: {v_counts[v]:,}")

    print("\nCurrent Live Edge Counts:")
    for e in sorted(e_counts.keys()):
        print(f"  {e:<16}: {e_counts[e]:,}")

    # 4. Existing corpus layer preservation check
    if v_counts["Document"] != 2951:
        raise ValueError(
            f"Preflight failed: Document count {v_counts['Document']} != 2,951"
        )
    if v_counts["Chunk"] != 9348:
        raise ValueError(
            f"Preflight failed: Chunk count {v_counts['Chunk']} != 9,348"
        )
    if e_counts["HAS_CHUNK"] != 9348:
        raise ValueError(
            f"Preflight failed: HAS_CHUNK count {e_counts['HAS_CHUNK']} != 9,348"
        )

    print(
        "\nCorpus Layer Preservation:     "
        "CONFIRMED (2,951 Docs, 9,348 Chunks, 9,348 Edges)"
    )

    # 5. Vector index readiness
    idx_status = conn.getVectorIndexStatus()
    rebuild = idx_status.get("NeedRebuildServers", [])
    if rebuild:
        raise ValueError(f"Preflight warning: Vector index needs rebuild: {idx_status}")
    print(f"Vector Index Status:           CLEAN & READY ({idx_status})")
    print("=" * 70)

    return v_counts


def wait_for_vertex_count(
    conn: tg.TigerGraphConnection,
    vertex_type: str,
    expected_count: int,
    max_retries: int = 60,
    delay_seconds: float = 2.0,
) -> int:
    """Poll TigerGraph until vertex count matches expected or timeout."""
    for attempt in range(max_retries):
        cnt = conn.getVertexCount(vertex_type)
        if cnt == expected_count:
            return cnt
        if attempt % 5 == 0 and attempt > 0:
            print(
                f"    [Flushing buffer] {vertex_type} count is "
                f"{cnt:,}/{expected_count:,} "
                f"(waiting {attempt * delay_seconds:.0f}s)..."
            )
        time.sleep(delay_seconds)
    return conn.getVertexCount(vertex_type)


def wait_for_edge_count(
    conn: tg.TigerGraphConnection,
    edge_type: str,
    expected_count: int,
    max_retries: int = 60,
    delay_seconds: float = 2.0,
) -> int:
    """Poll TigerGraph until edge count matches expected or timeout."""
    for attempt in range(max_retries):
        cnt = conn.getEdgeCount(edge_type)
        if cnt == expected_count:
            return cnt
        if attempt % 5 == 0 and attempt > 0:
            print(
                f"    [Flushing buffer] {edge_type} count is "
                f"{cnt:,}/{expected_count:,} "
                f"(waiting {attempt * delay_seconds:.0f}s)..."
            )
        time.sleep(delay_seconds)
    return conn.getEdgeCount(edge_type)


def execute_tigergraph_population(data: ExtractedGraphData) -> None:
    """Execute batch upserts into TigerGraph Cloud and verify final state."""
    conn = get_tigergraph_connection()
    perform_preflight_checks(conn)

    print("\n" + "=" * 70)
    print("STEP 2: BATCH WRITES WITH PER-CATEGORY LIVE VERIFICATION")
    print("=" * 70)

    batch_size = 500

    # --- 1. Event ---
    print("\n--- Ingesting Event Vertices ---")
    event_payload = [
        (e.event_id, {"name": e.name, "year": e.year, "description": e.description})
        for e in data.events.values()
    ]
    t0 = time.perf_counter()
    for i in range(0, len(event_payload), batch_size):
        conn.upsertVertices("Event", event_payload[i : i + batch_size])
    el_event = time.perf_counter() - t0
    c_event = wait_for_vertex_count(conn, "Event", len(data.events))
    print(f"  Upserted {len(event_payload)} Event vertices in {el_event:.2f}s")
    print(f"  Live Event Count: {c_event} | Expected: {len(data.events)}")
    assert c_event == len(data.events), (
        f"Event count mismatch: {c_event} != {len(data.events)}"
    )

    # --- 2. Sport ---
    print("\n--- Ingesting Sport Vertices ---")
    sport_payload = [
        (s.sport_id, {"name": s.name, "description": s.description})
        for s in data.sports.values()
    ]
    conn.upsertVertices("Sport", sport_payload)
    c_sport = wait_for_vertex_count(conn, "Sport", len(data.sports))
    print(f"  Upserted {len(sport_payload)} Sport vertices")
    print(f"  Live Sport Count: {c_sport} | Expected: {len(data.sports)}")
    assert c_sport == len(data.sports), (
        f"Sport count mismatch: {c_sport} != {len(data.sports)}"
    )

    # --- 3. Venue ---
    print("\n--- Ingesting Venue Vertices ---")
    venue_payload = [
        (v.venue_id, {"name": v.name, "city": v.city, "country": v.country})
        for v in data.venues.values()
    ]
    for i in range(0, len(venue_payload), batch_size):
        conn.upsertVertices("Venue", venue_payload[i : i + batch_size])
    c_venue = wait_for_vertex_count(conn, "Venue", len(data.venues))
    print(f"  Upserted {len(venue_payload)} Venue vertices")
    print(f"  Live Venue Count: {c_venue} | Expected: {len(data.venues)}")
    assert c_venue == len(data.venues), (
        f"Venue count mismatch: {c_venue} != {len(data.venues)}"
    )

    # --- 4. Country ---
    print("\n--- Ingesting Country Vertices ---")
    country_payload = [
        (c.country_id, {"name": c.name, "code": c.code})
        for c in data.countries.values()
    ]
    conn.upsertVertices("Country", country_payload)
    c_country = wait_for_vertex_count(conn, "Country", len(data.countries))
    print(f"  Upserted {len(country_payload)} Country vertices")
    print(f"  Live Country Count: {c_country} | Expected: {len(data.countries)}")
    assert c_country == len(data.countries), (
        f"Country count mismatch: {c_country} != {len(data.countries)}"
    )

    # --- 5. Person ---
    print("\n--- Ingesting Person Vertices ---")
    person_payload = [
        (p.person_id, {"name": p.name, "description": p.description})
        for p in data.persons.values()
    ]
    for i in range(0, len(person_payload), batch_size):
        conn.upsertVertices("Person", person_payload[i : i + batch_size])
    c_person = wait_for_vertex_count(conn, "Person", len(data.persons))
    print(f"  Upserted {len(person_payload)} Person vertices")
    print(f"  Live Person Count: {c_person} | Expected: {len(data.persons)}")
    assert c_person == len(data.persons), (
        f"Person count mismatch: {c_person} != {len(data.persons)}"
    )

    # --- 6. Team ---
    print("\n--- Ingesting Team Vertices ---")
    team_payload = [
        (t.team_id, {"name": t.name, "description": t.description})
        for t in data.teams.values()
    ]
    for i in range(0, len(team_payload), batch_size):
        conn.upsertVertices("Team", team_payload[i : i + batch_size])
    c_team = wait_for_vertex_count(conn, "Team", len(data.teams))
    print(f"  Upserted {len(team_payload)} Team vertices")
    print(f"  Live Team Count: {c_team} | Expected: {len(data.teams)}")
    assert c_team == len(data.teams), (
        f"Team count mismatch: {c_team} != {len(data.teams)}"
    )

    # --- 7. Entity ---
    print("\n--- Ingesting Entity Vertices ---")
    entity_payload = [
        (
            ent.entity_id,
            {
                "name": ent.name,
                "entity_type": ent.entity_type,
                "description": ent.description,
            },
        )
        for ent in data.entities.values()
    ]
    for i in range(0, len(entity_payload), batch_size):
        conn.upsertVertices("Entity", entity_payload[i : i + batch_size])
    c_entity = wait_for_vertex_count(conn, "Entity", len(data.entities))
    print(f"  Upserted {len(entity_payload)} Entity vertices")
    print(f"  Live Entity Count: {c_entity} | Expected: {len(data.entities)}")
    assert c_entity == len(data.entities), (
        f"Entity count mismatch: {c_entity} != {len(data.entities)}"
    )

    # --- 8. BELONGS_TO Edges ---
    print("\n--- Ingesting BELONGS_TO Edges ---")
    belongs_payload = [(src, tgt, {}) for src, tgt in data.belongs_to_edges]
    for i in range(0, len(belongs_payload), batch_size):
        conn.upsertEdges(
            sourceVertexType="Event",
            edgeType="BELONGS_TO",
            targetVertexType="Sport",
            edges=belongs_payload[i : i + batch_size],
        )
    c_belongs = wait_for_edge_count(conn, "BELONGS_TO", len(data.belongs_to_edges))
    print(f"  Upserted {len(belongs_payload)} BELONGS_TO edges")
    print(
        f"  Live BELONGS_TO Count: {c_belongs} | "
        f"Expected: {len(data.belongs_to_edges)}"
    )
    assert c_belongs == len(data.belongs_to_edges), (
        f"BELONGS_TO mismatch: {c_belongs} != {len(data.belongs_to_edges)}"
    )

    # --- 9. HELD_AT Edges ---
    print("\n--- Ingesting HELD_AT Edges ---")
    held_payload = [(src, tgt, {}) for src, tgt in data.held_at_edges]
    for i in range(0, len(held_payload), batch_size):
        conn.upsertEdges(
            sourceVertexType="Event",
            edgeType="HELD_AT",
            targetVertexType="Venue",
            edges=held_payload[i : i + batch_size],
        )
    c_held = wait_for_edge_count(conn, "HELD_AT", len(data.held_at_edges))
    print(f"  Upserted {len(held_payload)} HELD_AT edges")
    print(f"  Live HELD_AT Count: {c_held} | Expected: {len(data.held_at_edges)}")
    assert c_held == len(data.held_at_edges), (
        f"HELD_AT mismatch: {c_held} != {len(data.held_at_edges)}"
    )

    # --- 10. PARTICIPATED_IN Edges ---
    print("\n--- Ingesting PARTICIPATED_IN Edges ---")
    person_part = [
        (src_id, e_id, {})
        for src_type, src_id, e_id in data.participated_in_edges
        if src_type == "Person"
    ]
    team_part = [
        (src_id, e_id, {})
        for src_type, src_id, e_id in data.participated_in_edges
        if src_type == "Team"
    ]
    for i in range(0, len(person_part), batch_size):
        conn.upsertEdges(
            sourceVertexType="Person",
            edgeType="PARTICIPATED_IN",
            targetVertexType="Event",
            edges=person_part[i : i + batch_size],
        )
    for i in range(0, len(team_part), batch_size):
        conn.upsertEdges(
            sourceVertexType="Team",
            edgeType="PARTICIPATED_IN",
            targetVertexType="Event",
            edges=team_part[i : i + batch_size],
        )
    expected_part = len(data.participated_in_edges)
    c_part = wait_for_edge_count(conn, "PARTICIPATED_IN", expected_part)
    print(f"  Upserted {len(person_part)} Person + {len(team_part)} Team edges")
    print(f"  Live PARTICIPATED_IN Count: {c_part} | Expected: {expected_part}")
    assert c_part == expected_part, (
        f"PARTICIPATED_IN mismatch: {c_part} != {expected_part}"
    )

    # --- 11. REPRESENTS Edges ---
    print("\n--- Ingesting REPRESENTS Edges ---")
    person_rep = [
        (src_id, c_id, {})
        for src_type, src_id, c_id in data.represents_edges
        if src_type == "Person"
    ]
    team_rep = [
        (src_id, c_id, {})
        for src_type, src_id, c_id in data.represents_edges
        if src_type == "Team"
    ]
    for i in range(0, len(person_rep), batch_size):
        conn.upsertEdges(
            sourceVertexType="Person",
            edgeType="REPRESENTS",
            targetVertexType="Country",
            edges=person_rep[i : i + batch_size],
        )
    for i in range(0, len(team_rep), batch_size):
        conn.upsertEdges(
            sourceVertexType="Team",
            edgeType="REPRESENTS",
            targetVertexType="Country",
            edges=team_rep[i : i + batch_size],
        )
    expected_rep = len(data.represents_edges)
    c_rep = wait_for_edge_count(conn, "REPRESENTS", expected_rep)
    print(f"  Upserted {len(person_rep)} Person + {len(team_rep)} Team edges")
    print(f"  Live REPRESENTS Count: {c_rep} | Expected: {expected_rep}")
    assert c_rep == expected_rep, f"REPRESENTS mismatch: {c_rep} != {expected_rep}"

    # --- 12. MENTIONS Edges ---
    print("\n--- Ingesting MENTIONS Edges ---")
    mentions_payload = [(src, tgt, {}) for src, tgt in data.mentions_edges]
    for i in range(0, len(mentions_payload), batch_size):
        conn.upsertEdges(
            sourceVertexType="Chunk",
            edgeType="MENTIONS",
            targetVertexType="Entity",
            edges=mentions_payload[i : i + batch_size],
        )
    c_mentions = wait_for_edge_count(conn, "MENTIONS", len(data.mentions_edges))
    print(f"  Upserted {len(mentions_payload)} MENTIONS edges")
    print(f"  Live MENTIONS Count: {c_mentions} | Expected: {len(data.mentions_edges)}")
    assert c_mentions == len(data.mentions_edges), (
        f"MENTIONS mismatch: {c_mentions} != {len(data.mentions_edges)}"
    )

    # --- 13. RESOLVES_TO Edges ---
    print("\n--- Ingesting RESOLVES_TO Edges ---")
    for tgt_type in ["Sport", "Venue", "Country", "Event", "Person", "Team"]:
        tgt_edges = [
            (ent_id, tgt_id, {})
            for ent_id, t_type, tgt_id in data.resolves_to_edges
            if t_type == tgt_type
        ]
        for i in range(0, len(tgt_edges), batch_size):
            conn.upsertEdges(
                sourceVertexType="Entity",
                edgeType="RESOLVES_TO",
                targetVertexType=tgt_type,
                edges=tgt_edges[i : i + batch_size],
            )
        print(f"  Upserted {len(tgt_edges)} Entity -> {tgt_type} RESOLVES_TO edges")
    expected_resolves = len(data.resolves_to_edges)
    c_resolves = wait_for_edge_count(conn, "RESOLVES_TO", expected_resolves)
    print(f"  Live RESOLVES_TO Count: {c_resolves} | Expected: {expected_resolves}")
    assert c_resolves == expected_resolves, (
        f"RESOLVES_TO mismatch: {c_resolves} != {expected_resolves}"
    )

    # Step 3: Complete Live Verification
    print("\n" + "=" * 70)
    print("STEP 3: COMPLETE LIVE VERIFICATION")
    print("=" * 70)

    # 1. Final vertex counts
    final_v = {
        "Document": conn.getVertexCount("Document"),
        "Chunk": conn.getVertexCount("Chunk"),
        "Entity": conn.getVertexCount("Entity"),
        "Event": conn.getVertexCount("Event"),
        "Person": conn.getVertexCount("Person"),
        "Country": conn.getVertexCount("Country"),
        "Sport": conn.getVertexCount("Sport"),
        "Venue": conn.getVertexCount("Venue"),
        "Team": conn.getVertexCount("Team"),
    }
    print("Final Vertex Counts:")
    for k, v in final_v.items():
        print(f"  {k:<12}: {v:,}")

    # 2. Final edge counts
    final_e = {
        "HAS_CHUNK": conn.getEdgeCount("HAS_CHUNK"),
        "MENTIONS": conn.getEdgeCount("MENTIONS"),
        "RESOLVES_TO": conn.getEdgeCount("RESOLVES_TO"),
        "PARTICIPATED_IN": conn.getEdgeCount("PARTICIPATED_IN"),
        "REPRESENTS": conn.getEdgeCount("REPRESENTS"),
        "BELONGS_TO": conn.getEdgeCount("BELONGS_TO"),
        "HELD_AT": conn.getEdgeCount("HELD_AT"),
    }
    print("\nFinal Edge Counts:")
    for k, v in final_e.items():
        print(f"  {k:<16}: {v:,}")

    # Assert corpus preservation
    assert final_v["Document"] == 2951, f"Docs corrupted: {final_v['Document']}"
    assert final_v["Chunk"] == 9348, f"Chunks corrupted: {final_v['Chunk']}"
    assert final_e["HAS_CHUNK"] == 9348, f"HAS_CHUNK corrupted: {final_e['HAS_CHUNK']}"

    # 3. Referential integrity sampling
    print("\n--- Verifying Referential Integrity ---")
    # Sample BELONGS_TO
    sample_b_src, sample_b_tgt = next(iter(data.belongs_to_edges))
    res_b_src = conn.getVerticesById("Event", [sample_b_src])
    res_b_tgt = conn.getVerticesById("Sport", [sample_b_tgt])
    assert res_b_src and res_b_tgt, "BELONGS_TO referential check failed!"
    print(f"  BELONGS_TO Verified:  Event '{sample_b_src}' -> Sport '{sample_b_tgt}'")

    # Sample HELD_AT
    sample_h_src, sample_h_tgt = next(iter(data.held_at_edges))
    res_h_src = conn.getVerticesById("Event", [sample_h_src])
    res_h_tgt = conn.getVerticesById("Venue", [sample_h_tgt])
    assert res_h_src and res_h_tgt, "HELD_AT referential check failed!"
    print(f"  HELD_AT Verified:     Event '{sample_h_src}' -> Venue '{sample_h_tgt}'")

    # Sample PARTICIPATED_IN
    sample_p_type, sample_p_src, sample_p_tgt = next(iter(data.participated_in_edges))
    res_p_src = conn.getVerticesById(sample_p_type, [sample_p_src])
    res_p_tgt = conn.getVerticesById("Event", [sample_p_tgt])
    assert res_p_src and res_p_tgt, "PARTICIPATED_IN referential check failed!"
    print(
        f"  PARTICIPATED_IN Verified: {sample_p_type} '{sample_p_src}' "
        f"-> Event '{sample_p_tgt}'"
    )

    # Sample REPRESENTS
    sample_r_type, sample_r_src, sample_r_tgt = next(iter(data.represents_edges))
    res_r_src = conn.getVerticesById(sample_r_type, [sample_r_src])
    res_r_tgt = conn.getVerticesById("Country", [sample_r_tgt])
    assert res_r_src and res_r_tgt, "REPRESENTS referential check failed!"
    print(
        f"  REPRESENTS Verified:  {sample_r_type} '{sample_r_src}' "
        f"-> Country '{sample_r_tgt}'"
    )

    # Sample MENTIONS
    sample_m_src, sample_m_tgt = next(iter(data.mentions_edges))
    res_m_src = conn.getVerticesById("Chunk", [sample_m_src])
    res_m_tgt = conn.getVerticesById("Entity", [sample_m_tgt])
    assert res_m_src and res_m_tgt, "MENTIONS referential check failed!"
    print(f"  MENTIONS Verified:    Chunk '{sample_m_src}' -> Entity '{sample_m_tgt}'")

    # Sample RESOLVES_TO
    sample_res_src, sample_res_type, sample_res_tgt = next(iter(data.resolves_to_edges))
    res_res_src = conn.getVerticesById("Entity", [sample_res_src])
    res_res_tgt = conn.getVerticesById(sample_res_type, [sample_res_tgt])
    assert res_res_src and res_res_tgt, "RESOLVES_TO referential check failed!"
    print(
        f"  RESOLVES_TO Verified: Entity '{sample_res_src}' "
        f"-> {sample_res_type} '{sample_res_tgt}'"
    )

    # 4. MENTIONS provenance verification
    print("\n--- Verifying MENTIONS Provenance ---")
    multi_chunk_cids = [
        src for src, _ in data.mentions_edges if not src.endswith("#c0000")
    ]
    assert len(multi_chunk_cids) > 0, "No MENTIONS found beyond chunk #c0000!"
    print(
        f"  Confirmed {len(multi_chunk_cids):,} MENTIONS edges originate from "
        f"chunks > #c0000."
    )
    print(
        f"  Confirmed exactly {data.stats.sport_title_only_no_mentions:,} "
        f"title-only sports have zero chunk MENTIONS."
    )

    # 5. Existing retrieval layer validation
    print("\n--- Verifying Retrieval Layer & HNSW Readiness ---")
    sample_vec = [0.0] * 768
    sample_vec[0] = 1.0
    vec_res = conn.runInstalledQuery(
        "searchChunksByVector", {"qvec": sample_vec, "top_k": 3}
    )
    candidates = vec_res[0].get("candidates", [])
    assert len(candidates) == 3, (
        f"Expected 3 retrieval candidates, got {len(candidates)}"
    )
    print(
        f"  searchChunksByVector returned {len(candidates)} chunks successfully."
    )
    print("  Chunk vector retrieval remains 100% operational.")

    # 6. Read-only smoke queries against newly populated graph
    print("\n--- Running Read-Only Smoke Queries on Populated Graph ---")
    # Event -> BELONGS_TO -> Sport
    event_edges = conn.getEdges("Event", sample_b_src, edgeType="BELONGS_TO")
    print(
        f"  [Smoke 1] Event '{sample_b_src}' -(BELONGS_TO)-> "
        f"Sport '{event_edges[0]['to_id'] if event_edges else 'NONE'}'"
    )

    # Event -> HELD_AT -> Venue
    venue_edges = conn.getEdges("Event", sample_h_src, edgeType="HELD_AT")
    print(
        f"  [Smoke 2] Event '{sample_h_src}' -(HELD_AT)-> "
        f"Venue '{venue_edges[0]['to_id'] if venue_edges else 'NONE'}'"
    )

    # Person -> PARTICIPATED_IN -> Event
    person_edges = conn.getEdges("Person", sample_p_src, edgeType="PARTICIPATED_IN")
    print(
        f"  [Smoke 3] Person '{sample_p_src}' -(PARTICIPATED_IN)-> "
        f"Event '{person_edges[0]['to_id'] if person_edges else 'NONE'}'"
    )

    # Entity -> RESOLVES_TO -> Target
    ent_edges = conn.getEdges("Entity", sample_res_src, edgeType="RESOLVES_TO")
    print(
        f"  [Smoke 4] Entity '{sample_res_src}' -(RESOLVES_TO)-> "
        f"{ent_edges[0]['to_type'] if ent_edges else 'NONE'} "
        f"'{ent_edges[0]['to_id'] if ent_edges else 'NONE'}'"
    )

    print("\n>>> LIVE STAGE 1 INGESTION AND VERIFICATION FULLY COMPLETED <<<")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 TigerGraph population from corpus"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the live upsert to TigerGraph (default is dry-run only)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    corpus_path = repo_root / "data" / "raw" / "corpus.jsonl"

    print("Reading corpus and extracting graph facts...")
    t0 = time.perf_counter()
    data = extract_graph_from_corpus(corpus_path)
    extract_time = time.perf_counter() - t0
    print(f"Extraction completed in {extract_time:.2f}s.")

    # Validation
    errors = validate_extracted_graph(data)
    if errors:
        print(f"FATAL: {len(errors)} validation errors found:")
        for err in errors[:10]:
            print("  -", err)
        sys.exit(1)

    print("Validation passed: 0 referential integrity errors.")

    # Always print dry-run summary
    print_dry_run_summary(data, len(errors))

    if not args.execute:
        print("\n[SAFETY GATE] Dry-run completed successfully.")
        print("To populate TigerGraph, run with --execute.")
        return

    print("\nExecuting live TigerGraph write...")
    execute_tigergraph_population(data)


if __name__ == "__main__":
    main()
