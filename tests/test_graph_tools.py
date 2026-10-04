"""Unit tests for Stage 3 / A2 deterministic graph tools (mocked TigerGraph).

Covers all 17 required verification gates:
1. Person -> Events
2. Team -> Events
3. Event -> Sport/Venue
4. Person/Team -> Country
5. Venue -> Events
6. Year filtering
7. Sport filtering
8. Venue filtering
9. Event-name filtering
10. Distinct-country counting
11. Composite multi-hop filtering
12. Constraint intersection
13. Provenance/path preservation
14. Result bounds
15. Deterministic repeated execution
16. Failure handling
17. Trace generation
"""

from unittest.mock import MagicMock

from tgh.mcp.contracts import (
    CountryAggregation,
    CountryRepresentation,
    EventCandidate,
    EventContext,
    ToolExecutionResult,
)
from tgh.mcp.graph_tools import GraphToolBounds, TigerGraphTools
from tgh.telemetry.trace import TraceRecorder


def build_mock_connection() -> MagicMock:
    """Construct mock TigerGraph connection with simulated Olympic graph data."""
    mock_conn = MagicMock()

    # Pre-canned vertices
    vertices = {
        ("Event", "Q25239316"): {
            "v_id": "Q25239316",
            "v_type": "Event",
            "attributes": {
                "event_id": "Q25239316",
                "name": "Men's 60 kg",
                "year": 1988,
                "description": (
                    "Weightlifting at the 1988 Summer Olympics – Men's 60 kg"
                ),
            },
        },
        ("Event", "Q25239317"): {
            "v_id": "Q25239317",
            "v_type": "Event",
            "attributes": {
                "event_id": "Q25239317",
                "name": "Men's +110 kg",
                "year": 1988,
                "description": (
                    "Weightlifting at the 1988 Summer Olympics – Men's +110 kg"
                ),
            },
        },
        ("Event", "Q25301483"): {
            "v_id": "Q25301483",
            "v_type": "Event",
            "attributes": {
                "event_id": "Q25301483",
                "name": "Women's singles",
                "year": 2016,
                "description": (
                    "Badminton at the 2016 Summer Olympics – Women's singles"
                ),
            },
        },
        ("Event", "Q25301472"): {
            "v_id": "Q25301472",
            "v_type": "Event",
            "attributes": {
                "event_id": "Q25301472",
                "name": "Men's singles",
                "year": 2016,
                "description": "Badminton at the 2016 Summer Olympics – Men's singles",
            },
        },
        ("Sport", "sport_weightlifting"): {
            "v_id": "sport_weightlifting",
            "v_type": "Sport",
            "attributes": {"name": "Weightlifting"},
        },
        ("Sport", "sport_badminton"): {
            "v_id": "sport_badminton",
            "v_type": "Sport",
            "attributes": {"name": "Badminton"},
        },
        ("Venue", "venue_olympic_weightlifting_gymnasium"): {
            "v_id": "venue_olympic_weightlifting_gymnasium",
            "v_type": "Venue",
            "attributes": {"name": "Olympic Weightlifting Gymnasium"},
        },
        ("Venue", "venue_riocentro_pavilion_4"): {
            "v_id": "venue_riocentro_pavilion_4",
            "v_type": "Venue",
            "attributes": {"name": "Riocentro - Pavilion 4"},
        },
        ("Person", "person_naim_suleymanoglu"): {
            "v_id": "person_naim_suleymanoglu",
            "v_type": "Person",
            "attributes": {"name": "Naim Süleymanoğlu"},
        },
        ("Person", "person_carolina_marin"): {
            "v_id": "person_carolina_marin",
            "v_type": "Person",
            "attributes": {"name": "Carolina Marín"},
        },
        ("Team", "team_spain_badminton"): {
            "v_id": "team_spain_badminton",
            "v_type": "Team",
            "attributes": {"name": "Spain Badminton Squad"},
        },
        ("Country", "country_TUR"): {
            "v_id": "country_TUR",
            "v_type": "Country",
            "attributes": {"name": "Turkey", "code": "TUR"},
        },
        ("Country", "country_ESP"): {
            "v_id": "country_ESP",
            "v_type": "Country",
            "attributes": {"name": "Spain", "code": "ESP"},
        },
    }

    # Pre-canned edges
    edges = {
        ("Person", "person_naim_suleymanoglu", "PARTICIPATED_IN"): [
            {"to_id": "Q25239316", "to_type": "Event"}
        ],
        ("Person", "person_naim_suleymanoglu", "REPRESENTS"): [
            {"to_id": "country_TUR", "to_type": "Country"}
        ],
        ("Team", "team_spain_badminton", "PARTICIPATED_IN"): [
            {"to_id": "Q25301483", "to_type": "Event"}
        ],
        ("Team", "team_spain_badminton", "REPRESENTS"): [
            {"to_id": "country_ESP", "to_type": "Country"}
        ],
        ("Person", "person_carolina_marin", "PARTICIPATED_IN"): [
            {"to_id": "Q25301483", "to_type": "Event"}
        ],
        ("Person", "person_carolina_marin", "REPRESENTS"): [
            {"to_id": "country_ESP", "to_type": "Country"}
        ],
        ("Venue", "venue_olympic_weightlifting_gymnasium", "reverse_HELD_AT"): [
            {"to_id": "Q25239316", "to_type": "Event"},
            {"to_id": "Q25239317", "to_type": "Event"},
        ],
        ("Venue", "venue_riocentro_pavilion_4", "reverse_HELD_AT"): [
            {"to_id": "Q25301483", "to_type": "Event"},
            {"to_id": "Q25301472", "to_type": "Event"},
        ],
        ("Event", "Q25239316", ""): [
            {
                "e_type": "BELONGS_TO",
                "to_type": "Sport",
                "to_id": "sport_weightlifting",
            },
            {
                "e_type": "HELD_AT",
                "to_type": "Venue",
                "to_id": "venue_olympic_weightlifting_gymnasium",
            },
            {
                "e_type": "reverse_PARTICIPATED_IN",
                "to_type": "Person",
                "to_id": "person_naim_suleymanoglu",
            },
        ],
        ("Event", "Q25239317", ""): [
            {
                "e_type": "BELONGS_TO",
                "to_type": "Sport",
                "to_id": "sport_weightlifting",
            },
            {
                "e_type": "HELD_AT",
                "to_type": "Venue",
                "to_id": "venue_olympic_weightlifting_gymnasium",
            },
        ],
        ("Event", "Q25301483", ""): [
            {"e_type": "BELONGS_TO", "to_type": "Sport", "to_id": "sport_badminton"},
            {
                "e_type": "HELD_AT",
                "to_type": "Venue",
                "to_id": "venue_riocentro_pavilion_4",
            },
            {
                "e_type": "reverse_PARTICIPATED_IN",
                "to_type": "Person",
                "to_id": "person_carolina_marin",
            },
            {
                "e_type": "reverse_PARTICIPATED_IN",
                "to_type": "Team",
                "to_id": "team_spain_badminton",
            },
        ],
        ("Event", "Q25239316", "BELONGS_TO"): [
            {"e_type": "BELONGS_TO", "to_type": "Sport", "to_id": "sport_weightlifting"}
        ],
        ("Event", "Q25239316", "HELD_AT"): [
            {
                "e_type": "HELD_AT",
                "to_type": "Venue",
                "to_id": "venue_olympic_weightlifting_gymnasium",
            }
        ],
        ("Event", "Q25301483", "BELONGS_TO"): [
            {"e_type": "BELONGS_TO", "to_type": "Sport", "to_id": "sport_badminton"}
        ],
        ("Event", "Q25301483", "HELD_AT"): [
            {
                "e_type": "HELD_AT",
                "to_type": "Venue",
                "to_id": "venue_riocentro_pavilion_4",
            }
        ],
        ("Event", "Q25301472", "BELONGS_TO"): [
            {"e_type": "BELONGS_TO", "to_type": "Sport", "to_id": "sport_badminton"}
        ],
        ("Event", "Q25301472", "HELD_AT"): [
            {
                "e_type": "HELD_AT",
                "to_type": "Venue",
                "to_id": "venue_riocentro_pavilion_4",
            }
        ],
    }

    def fake_get_vertices(v_type: str, ids: list[str]) -> list[dict]:
        return [vertices[(v_type, vid)] for vid in ids if (v_type, vid) in vertices]

    def fake_get_edges(v_type: str, vid: str, edgeType: str = "") -> list[dict]:
        return edges.get((v_type, vid, edgeType), [])

    mock_conn.getVerticesById.side_effect = fake_get_vertices
    mock_conn.getEdges.side_effect = fake_get_edges
    return mock_conn


# -----------------------------------------------------------------------------
# Tests 1-17
# -----------------------------------------------------------------------------


def test_person_to_events():
    """1. Test Person -> PARTICIPATED_IN -> Event candidates."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.get_events_for_participant("Person", "person_naim_suleymanoglu")
    assert res.success is True
    assert len(res.data) == 1
    ev = res.data[0]
    assert isinstance(ev, EventCandidate)
    assert ev.event_id == "Q25239316"
    assert ev.year == 1988
    assert "PARTICIPATED_IN" in ev.provenance_path


def test_team_to_events():
    """2. Test Team -> PARTICIPATED_IN -> Event candidates."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.get_events_for_participant("Team", "team_spain_badminton")
    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0].event_id == "Q25301483"


def test_event_to_sport_and_venue():
    """3. Test Event -> Sport/Venue and reverse participants."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.get_event_context("Q25239316")
    assert res.success is True
    ctx = res.data
    assert isinstance(ctx, EventContext)
    assert ctx.sport["sport_id"] == "sport_weightlifting"
    assert ctx.venue["venue_id"] == "venue_olympic_weightlifting_gymnasium"
    assert len(ctx.participants) == 1
    assert ctx.participants[0]["id"] == "person_naim_suleymanoglu"


def test_person_team_to_country():
    """4. Test Person/Team -> REPRESENTS -> Country."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    p_res = tools.get_country_for_participant("Person", "person_naim_suleymanoglu")
    assert p_res.success is True
    assert len(p_res.data) == 1
    rep = p_res.data[0]
    assert isinstance(rep, CountryRepresentation)
    assert rep.country_id == "country_TUR"
    assert rep.country_code == "TUR"

    t_res = tools.get_country_for_participant("Team", "team_spain_badminton")
    assert t_res.success is True
    assert t_res.data[0].country_id == "country_ESP"


def test_venue_to_events():
    """5. Test Venue -> reverse_HELD_AT -> Event candidates."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.get_events_for_venue("venue_olympic_weightlifting_gymnasium")
    assert res.success is True
    assert len(res.data) == 2
    eids = [e.event_id for e in res.data]
    assert "Q25239316" in eids
    assert "Q25239317" in eids


def test_year_filtering():
    """6. Test deterministic year filtering."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res_1988 = tools.filter_events(["Q25239316", "Q25301483"], year=1988)
    assert res_1988.success is True
    assert len(res_1988.data) == 1
    assert res_1988.data[0].event_id == "Q25239316"

    res_2016 = tools.filter_events(["Q25239316", "Q25301483"], year=2016)
    assert len(res_2016.data) == 1
    assert res_2016.data[0].event_id == "Q25301483"


def test_sport_filtering():
    """7. Test deterministic sport filtering."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.filter_events(["Q25239316", "Q25301483"], sport="badminton")
    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0].event_id == "Q25301483"


def test_venue_filtering():
    """8. Test deterministic venue filtering."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.filter_events(
        ["Q25239316", "Q25301483"],
        venue_id="venue_olympic_weightlifting_gymnasium",
    )
    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0].event_id == "Q25239316"


def test_event_name_filtering():
    """9. Test deterministic event-name filtering (substring & normalization)."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.filter_events(
        ["Q25301483", "Q25301472"],
        event_name_query="Women's singles",
    )
    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0].event_id == "Q25301483"


def test_distinct_country_counting():
    """10. Test distinct country aggregation and deduplication."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    participants = [
        {"id": "person_carolina_marin", "type": "Person"},
        {"id": "team_spain_badminton", "type": "Team"},
        {"id": "person_naim_suleymanoglu", "type": "Person"},
    ]
    res = tools.aggregate_countries_for_participants(participants)
    assert res.success is True
    agg = res.data
    assert isinstance(agg, CountryAggregation)
    assert agg.distinct_count == 2
    assert "country_ESP" in agg.country_ids
    assert "country_TUR" in agg.country_ids


def test_composite_multihop_filtering():
    """11. Test composite multi-hop: venue -> event -> participant."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.execute_composite_multihop(
        venue_id="venue_riocentro_pavilion_4",
        year=2016,
        sport="badminton",
        event_name_query="Women's singles",
    )
    assert res.success is True
    events = res.data["events"]
    assert len(events) == 1
    assert events[0].event_id == "Q25301483"
    ctxs = res.data["event_contexts"]
    assert len(ctxs) == 1
    assert ctxs[0].participants[0]["id"] == "person_carolina_marin"


def test_constraint_intersection():
    """12. Test intersecting participant and venue constraints."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    # Carolina Marin at Riocentro Pavilion 4
    res = tools.execute_composite_multihop(
        participant_type="Person",
        participant_id="person_carolina_marin",
        venue_id="venue_riocentro_pavilion_4",
    )
    assert res.success is True
    assert len(res.data["events"]) == 1
    assert res.data["events"][0].event_id == "Q25301483"

    # Carolina Marin at Weightlifting Gymnasium -> 0 intersection
    empty_res = tools.execute_composite_multihop(
        participant_type="Person",
        participant_id="person_carolina_marin",
        venue_id="venue_olympic_weightlifting_gymnasium",
    )
    assert empty_res.success is True
    assert len(empty_res.data["events"]) == 0


def test_provenance_preservation():
    """13. Test provenance path generation and preservation."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.execute_composite_multihop(
        venue_id="venue_olympic_weightlifting_gymnasium",
        year=1988,
        event_name_query="60 kg",
    )
    assert res.success is True
    assert len(res.provenance) > 0
    # Must preserve graph edge labels
    has_held_at = any("HELD_AT" in p for p in res.provenance)
    assert has_held_at is True


def test_result_bounds():
    """14. Test safety limits on returned candidate lists."""
    conn = build_mock_connection()
    bounds = GraphToolBounds(max_events=1, max_participants=1, max_countries=1)
    tools = TigerGraphTools(conn=conn, bounds=bounds)

    res = tools.get_events_for_venue("venue_olympic_weightlifting_gymnasium")
    assert res.success is True
    assert len(res.data) <= 1


def test_deterministic_repeated_execution():
    """15. Test identical execution across repeated calls."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res1 = tools.execute_composite_multihop(
        venue_id="venue_riocentro_pavilion_4",
        year=2016,
    )
    res2 = tools.execute_composite_multihop(
        venue_id="venue_riocentro_pavilion_4",
        year=2016,
    )
    assert [e.event_id for e in res1.data["events"]] == [
        e.event_id for e in res2.data["events"]
    ]


def test_failure_handling():
    """16. Test graceful structured error response on invalid input."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    res = tools.get_events_for_participant("AlienType", "p123")
    assert isinstance(res, ToolExecutionResult)
    assert res.success is False
    assert "Invalid participant_type" in (res.error or "")


def test_trace_generation():
    """17. Test structured trace logging and latency recording."""
    conn = build_mock_connection()
    recorder = TraceRecorder(run_id="test-run-123")
    tools = TigerGraphTools(conn=conn, recorder=recorder)

    tools.get_event_context("Q25239316")
    traces = recorder.get_traces()
    assert len(traces) == 1
    t = traces[0]
    assert t.run_id == "test-run-123"
    assert t.operation_name == "get_event_context"
    assert t.success is True
    assert t.latency_ms >= 0.0
    assert "event_id" in t.result_summary


def test_venue_prefiltered_no_redundant_edge_lookup():
    """18. Test venue-prefiltered candidate events skip redundant getEdges."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    # When venue_prefiltered=True and no sport filter,
    # conn.getEdges shouldn't be called for Event
    conn.getEdges.reset_mock()
    res = tools.filter_events(
        ["Q25239316"],
        year=1988,
        venue_id="venue_olympic_weightlifting_gymnasium",
        venue_prefiltered=True,
    )
    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0].event_id == "Q25239316"
    assert "Venue:venue_olympic_weightlifting_gymnasium" in res.data[0].provenance_path
    # Confirm getEdges was not called for Event
    assert conn.getEdges.call_count == 0


def test_venue_prefiltered_with_sport_filter_preserves_filtering():
    """19. Test venue-prefiltered candidate events with sport still inspect edges."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    conn.getEdges.reset_mock()
    res = tools.filter_events(
        ["Q25239316", "Q25301483"],
        venue_id="venue_olympic_weightlifting_gymnasium",
        sport="weightlifting",
        venue_prefiltered=True,
    )
    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0].event_id == "Q25239316"
    # Sport filter required edge lookup
    assert conn.getEdges.call_count > 0


def test_filter_events_default_preserves_unverified_venue_filtering():
    """20. Test non-venue-prefiltered events still verify venue relationship."""
    conn = build_mock_connection()
    tools = TigerGraphTools(conn=conn)

    conn.getEdges.reset_mock()
    res = tools.filter_events(
        ["Q25239316", "Q25301483"],
        venue_id="venue_olympic_weightlifting_gymnasium",
        venue_prefiltered=False,
    )
    assert res.success is True
    assert len(res.data) == 1
    assert res.data[0].event_id == "Q25239316"
    # Unverified venue filter required edge lookup
    assert conn.getEdges.call_count > 0

