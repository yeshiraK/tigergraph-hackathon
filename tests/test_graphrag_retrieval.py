"""Unit tests for deterministic GraphRAG retrieval (mocked TigerGraph).

Tests cover:
1. Seed retrieval result parsing
2. Entity deduplication
3. Domain vertex deduplication
4. Graph path representation
5. Chunk deduplication
6. Bounded expansion
7. Maximum evidence limits
8. Deterministic ranking
9. Provenance preservation
10. No unrestricted recursive traversal
"""

from unittest.mock import MagicMock

import pytest

from tgh.retrieval.graph import GraphExpansionConfig, TigerGraphTraverser
from tgh.retrieval.graphrag import GraphRAGRetriever
from tgh.retrieval.models import (
    EvidenceChunk,
    GraphPath,
    SeedChunk,
)
from tgh.retrieval.vector import TigerGraphVectorRetriever


def test_seed_retrieval_parsing():
    """1. Test seed retrieval result parsing from TigerGraph query response."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {
                    "v_id": "Q100#c0000",
                    "v_type": "Chunk",
                    "attributes": {
                        "candidates.text": "Sample text for chunk 100",
                    },
                },
                {
                    "v_id": "Q200#c0001",
                    "v_type": "Chunk",
                    "attributes": {
                        "candidates.text": "Sample text for chunk 200",
                    },
                },
            ],
            "@@distances": {
                "Q100#c0000": 0.25,
                "Q200#c0001": 0.15,
            },
            "@@chunk_to_doc": {
                "Q100#c0000": "Q100",
                "Q200#c0001": "Q200",
            },
        }
    ]

    mock_provider = MagicMock()
    mock_provider.format_query.return_value = "search_query: test"
    mock_provider.embed_text.return_value = [0.1] * 768

    retriever = TigerGraphVectorRetriever(conn=mock_conn, provider=mock_provider)
    seeds = retriever.retrieve_seeds("test question", top_k=2)

    assert len(seeds) == 2
    # Sorted by ascending distance: Q200#c0001 (0.15) should be rank 1
    assert seeds[0].chunk_id == "Q200#c0001"
    assert seeds[0].doc_id == "Q200"
    assert seeds[0].distance == pytest.approx(0.15)
    assert seeds[0].similarity == pytest.approx(0.85)
    assert seeds[0].rank == 1

    assert seeds[1].chunk_id == "Q100#c0000"
    assert seeds[1].doc_id == "Q100"
    assert seeds[1].distance == pytest.approx(0.25)
    assert seeds[1].similarity == pytest.approx(0.75)
    assert seeds[1].rank == 2


def test_entity_deduplication():
    """2. Test that entities discovered across seeds are deduplicated."""
    mock_conn = MagicMock()

    # Both seeds mention the same entity_country_USA
    def mock_get_edges(v_type, v_id, edgeType=""):
        if v_type == "Chunk" and edgeType == "MENTIONS":
            return [{"to_type": "Entity", "to_id": "entity_country_USA"}]
        if v_type == "Entity" and edgeType == "RESOLVES_TO":
            return [{"to_type": "Country", "to_id": "country_USA"}]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges

    seeds = [
        SeedChunk("Q1#c0000", "Q1", distance=0.1, similarity=0.9, rank=1),
        SeedChunk("Q2#c0000", "Q2", distance=0.2, similarity=0.8, rank=2),
    ]

    traverser = TigerGraphTraverser(conn=mock_conn)
    ents, doms, paths, chunks = traverser.expand_seeds(seeds)

    # entity_country_USA should only appear once in discovered_entities
    ent_ids = [e.entity_id for e in ents]
    assert ent_ids == ["entity_country_USA"]


def test_domain_vertex_deduplication():
    """3. Test that domain vertices reached via different entities are deduplicated."""
    mock_conn = MagicMock()

    def mock_get_edges(v_type, v_id, edgeType=""):
        if v_type == "Chunk":
            return [
                {"to_type": "Entity", "to_id": "entity_1"},
                {"to_type": "Entity", "to_id": "entity_2"},
            ]
        if v_type == "Entity" and edgeType == "RESOLVES_TO":
            # Both entities resolve to the same person
            return [{"to_type": "Person", "to_id": "person_simone_biles"}]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges

    seeds = [SeedChunk("Q1#c0000", "Q1", distance=0.1, similarity=0.9, rank=1)]
    traverser = TigerGraphTraverser(conn=mock_conn)
    ents, doms, paths, chunks = traverser.expand_seeds(seeds)

    assert len(doms) == 1
    assert doms[0].vertex_type == "Person"
    assert doms[0].vertex_id == "person_simone_biles"


def test_graph_path_representation():
    """4. Test GraphPath string representation and serialization."""
    gp = GraphPath(
        source_type="Person",
        source_id="person_bolt",
        edge_type="PARTICIPATED_IN",
        target_type="Event",
        target_id="event_100m",
        hop=2,
    )
    expected_str = "(Person:person_bolt) -[PARTICIPATED_IN]-> (Event:event_100m)"
    assert gp.to_str() == expected_str
    d = gp.to_dict()
    assert d["source_id"] == "person_bolt"
    assert d["hop"] == 2


def test_chunk_deduplication():
    """5. Test that derived chunks are deduplicated across expansion paths."""
    mock_conn = MagicMock()

    def mock_get_edges(v_type, v_id, edgeType=""):
        if v_type == "Chunk":
            return [{"to_type": "Entity", "to_id": "ent_p1"}]
        if v_type == "Entity" and edgeType == "RESOLVES_TO":
            return [{"to_type": "Person", "to_id": "p1"}]
        if v_type == "Person" and edgeType == "PARTICIPATED_IN":
            # Resolves to event EV1
            return [{"to_type": "Event", "to_id": "EV1"}]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges

    # Two seeds expanding to the same event EV1
    seeds = [
        SeedChunk("Q1#c0000", "Q1", distance=0.2, similarity=0.8, rank=1),
        SeedChunk("Q2#c0000", "Q2", distance=0.1, similarity=0.9, rank=2),
    ]

    traverser = TigerGraphTraverser(conn=mock_conn)
    ents, doms, paths, chunks = traverser.expand_seeds(seeds)

    chunk_ids = [c.chunk_id for c in chunks]
    assert chunk_ids == ["EV1#c0000"]
    # Best score from seed Q2 (0.9 * decay) should be retained
    assert chunks[0].score == pytest.approx(0.9 * (traverser.config.decay_per_hop**2))


def test_bounded_expansion():
    """6. Test that expansion obeys maximum limits on entities and events."""
    mock_conn = MagicMock()

    def mock_get_edges(v_type, v_id, edgeType=""):
        if v_type == "Chunk":
            # Returns 10 entities
            return [{"to_type": "Entity", "to_id": f"ent_{i}"} for i in range(10)]
        if v_type == "Entity" and edgeType == "RESOLVES_TO":
            return [{"to_type": "Venue", "to_id": "venue_1"}]
        if v_type == "Venue" and edgeType == "reverse_HELD_AT":
            # Returns 20 events
            return [{"to_type": "Event", "to_id": f"ev_{i}"} for i in range(20)]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges

    cfg = GraphExpansionConfig(
        max_entities_per_seed=3,
        max_events_per_venue=5,
    )
    seeds = [SeedChunk("Q1#c0000", "Q1", distance=0.1, similarity=0.9, rank=1)]
    traverser = TigerGraphTraverser(conn=mock_conn, config=cfg)
    ents, doms, paths, chunks = traverser.expand_seeds(seeds)

    # Ent limit was 3
    assert len(ents) == 3
    # Venue limit was 5 events
    assert len(chunks) <= 5


def test_maximum_evidence_limits():
    """7. Test that total graph chunks do not exceed max_graph_derived_chunks."""
    mock_conn = MagicMock()

    def mock_get_edges(v_type, v_id, edgeType=""):
        if v_type == "Chunk":
            return [{"to_type": "Entity", "to_id": "ent_1"}]
        if v_type == "Entity" and edgeType == "RESOLVES_TO":
            return [{"to_type": "Venue", "to_id": "v1"}]
        if v_type == "Venue":
            return [{"to_type": "Event", "to_id": f"ev_{i}"} for i in range(50)]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges

    cfg = GraphExpansionConfig(
        max_events_per_venue=50,
        max_graph_derived_chunks=7,
    )
    seeds = [SeedChunk("Q1#c0000", "Q1", distance=0.1, similarity=0.9, rank=1)]
    traverser = TigerGraphTraverser(conn=mock_conn, config=cfg)
    ents, doms, paths, chunks = traverser.expand_seeds(seeds)

    assert len(chunks) == 7


def test_deterministic_ranking():
    """8. Test deterministic evidence fusion and document ranking."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {"v_id": "Q1#c0000", "attributes": {"candidates.text": "text1"}},
                {"v_id": "Q2#c0000", "attributes": {"candidates.text": "text2"}},
            ],
            "@@distances": {"Q1#c0000": 0.2, "Q2#c0000": 0.3},
            "@@chunk_to_doc": {"Q1#c0000": "Q1", "Q2#c0000": "Q2"},
        }
    ]

    mock_provider = MagicMock()
    mock_provider.format_query.return_value = "query"
    mock_provider.embed_text.return_value = [0.0] * 768

    retriever = GraphRAGRetriever(conn=mock_conn, embedding_provider=mock_provider)

    # Mock fusion directly
    seeds = [
        SeedChunk("Q1#c0000", "Q1", distance=0.2, similarity=0.8, rank=1),
        SeedChunk("Q2#c0000", "Q2", distance=0.3, similarity=0.7, rank=2),
    ]
    graph_chunks = [
        EvidenceChunk("Q3#c0000", "Q3", score=0.75, source="graph", hop_distance=1),
    ]

    fused = retriever._fuse_evidence(seeds, graph_chunks)
    assert len(fused) == 3
    # Order should be Q1 (0.8), Q3 (0.75), Q2 (0.7)
    assert [c.chunk_id for c in fused] == ["Q1#c0000", "Q3#c0000", "Q2#c0000"]

    docs = retriever._rank_documents(fused)
    assert docs == ["Q1", "Q3", "Q2"]


def test_provenance_preservation():
    """9. Test that provenance explains why evidence chunks were included."""
    mock_conn = MagicMock()

    def mock_get_edges(v_type, v_id, edgeType=""):
        if v_type == "Chunk":
            return [{"to_type": "Entity", "to_id": "entity_person_michael_phelps"}]
        if v_type == "Entity" and edgeType == "RESOLVES_TO":
            return [{"to_type": "Person", "to_id": "person_michael_phelps"}]
        if v_type == "Person" and edgeType == "PARTICIPATED_IN":
            return [{"to_type": "Event", "to_id": "Q12345"}]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges

    seeds = [SeedChunk("Q1#c0000", "Q1", distance=0.2, similarity=0.8, rank=1)]
    traverser = TigerGraphTraverser(conn=mock_conn)
    ents, doms, paths, chunks = traverser.expand_seeds(seeds)

    assert len(chunks) == 1
    ev_chunk = chunks[0]
    assert len(ev_chunk.provenance) > 0
    prov_text = ev_chunk.provenance[0]
    assert "entity_person_michael_phelps" in prov_text
    assert "person_michael_phelps" in prov_text
    assert "Event Q12345" in prov_text
    assert "Chunk Q12345#c0000" in prov_text


def test_no_unrestricted_recursive_traversal():
    """10. Test that expansion strictly terminates at max_hops and does not loop."""
    mock_conn = MagicMock()

    call_count = 0

    def mock_get_edges(v_type, v_id, edgeType=""):
        nonlocal call_count
        call_count += 1
        # Cyclic mock: Chunk -> Entity -> Person -> Event -> Entity ...
        if v_type == "Chunk":
            return [{"to_type": "Entity", "to_id": "ent_cyclic"}]
        if v_type == "Entity":
            return [{"to_type": "Person", "to_id": "person_cyclic"}]
        if v_type == "Person":
            return [{"to_type": "Event", "to_id": "event_cyclic"}]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges

    cfg = GraphExpansionConfig(max_hops=2)
    seeds = [SeedChunk("Q1#c0000", "Q1", distance=0.1, similarity=0.9, rank=1)]
    traverser = TigerGraphTraverser(conn=mock_conn, config=cfg)
    ents, doms, paths, chunks = traverser.expand_seeds(seeds)

    # Must terminate without infinite recursion, bounded calls
    assert call_count < 20
    assert len(chunks) == 1
    assert chunks[0].hop_distance <= 2


def test_adaptive_gating_confident_non_relational():
    """11. Test that confident non-relational queries preserve vector-only ranking."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {"v_id": "Q1#c0000", "attributes": {"candidates.text": "text1"}},
                {"v_id": "Q2#c0000", "attributes": {"candidates.text": "text2"}},
            ],
            "@@distances": {"Q1#c0000": 0.12, "Q2#c0000": 0.25},
            "@@chunk_to_doc": {"Q1#c0000": "Q1", "Q2#c0000": "Q2"},
        }
    ]
    mock_provider = MagicMock()
    mock_provider.format_query.return_value = "query"
    mock_provider.embed_text.return_value = [0.0] * 768

    retriever = GraphRAGRetriever(conn=mock_conn, embedding_provider=mock_provider)
    res = retriever.retrieve("How many nations competed in the women's 100m?")

    # Gating should be bypassed (confident: sim 0.88, margin 0.13)
    assert not res.is_graph_expanded
    assert "confident_vector" in res.decision_reason
    assert res.ranked_doc_ids == ["Q1", "Q2"]
    # getEdges should never have been called
    mock_conn.getEdges.assert_not_called()


def test_adaptive_gating_relational_query():
    """12. Test that relational multi-hop queries trigger graph expansion."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {"v_id": "Q1#c0000", "attributes": {"candidates.text": "venue text"}},
            ],
            "@@distances": {"Q1#c0000": 0.15},
            "@@chunk_to_doc": {"Q1#c0000": "Q1"},
        }
    ]
    mock_conn.getEdges.return_value = []
    mock_provider = MagicMock()
    mock_provider.format_query.return_value = "query"
    mock_provider.embed_text.return_value = [0.0] * 768

    retriever = GraphRAGRetriever(conn=mock_conn, embedding_provider=mock_provider)
    q = "Who won the gold medal in the event held at London Velopark on 4 August?"
    res = retriever.retrieve(q)

    assert res.is_graph_expanded
    assert "relational_language" in res.decision_reason
    mock_conn.getEdges.assert_called()


def test_adaptive_gating_weak_vector_retrieval():
    """13. Test that weak vector confidence triggers graph expansion."""
    mock_conn = MagicMock()
    # Low similarity (distance 0.35 => similarity 0.65 < 0.72)
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {"v_id": "Q1#c0000", "attributes": {"candidates.text": "weak text"}},
            ],
            "@@distances": {"Q1#c0000": 0.35},
            "@@chunk_to_doc": {"Q1#c0000": "Q1"},
        }
    ]
    mock_conn.getEdges.return_value = []
    mock_provider = MagicMock()
    mock_provider.format_query.return_value = "query"
    mock_provider.embed_text.return_value = [0.0] * 768

    retriever = GraphRAGRetriever(conn=mock_conn, embedding_provider=mock_provider)
    res = retriever.retrieve("Generic question without relational phrases")

    assert res.is_graph_expanded
    assert "low_vector_confidence" in res.decision_reason


def test_adaptive_deterministic_behavior():
    """14. Test deterministic output across multiple identical calls."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {"v_id": "Q1#c0000", "attributes": {"candidates.text": "t1"}},
                {"v_id": "Q2#c0000", "attributes": {"candidates.text": "t2"}},
            ],
            "@@distances": {"Q1#c0000": 0.10, "Q2#c0000": 0.20},
            "@@chunk_to_doc": {"Q1#c0000": "Q1", "Q2#c0000": "Q2"},
        }
    ]
    mock_provider = MagicMock()
    mock_provider.format_query.return_value = "query"
    mock_provider.embed_text.return_value = [0.0] * 768

    retriever = GraphRAGRetriever(conn=mock_conn, embedding_provider=mock_provider)
    res1 = retriever.retrieve("How many competitors were in archery?")
    res2 = retriever.retrieve("How many competitors were in archery?")

    assert res1.ranked_doc_ids == res2.ranked_doc_ids
    assert res1.is_graph_expanded == res2.is_graph_expanded
    assert res1.decision_reason == res2.decision_reason
    assert [c.score for c in res1.evidence_chunks] == [
        c.score for c in res2.evidence_chunks
    ]


def test_adaptive_provenance_intact():
    """15. Test that provenance is preserved in both bypassed and expanded modes."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {"v_id": "Q1#c0000", "attributes": {"candidates.text": "t1"}},
            ],
            "@@distances": {"Q1#c0000": 0.15},
            "@@chunk_to_doc": {"Q1#c0000": "Q1"},
        }
    ]
    mock_provider = MagicMock()
    mock_provider.format_query.return_value = "query"
    mock_provider.embed_text.return_value = [0.0] * 768

    retriever = GraphRAGRetriever(conn=mock_conn, embedding_provider=mock_provider)

    # 1. Bypassed mode
    bypassed_res = retriever.retrieve("How many athletes participated?")
    assert len(bypassed_res.evidence_chunks) == 1
    assert "VectorSearch" in bypassed_res.evidence_chunks[0].provenance[0]

    # 2. Expanded mode
    mock_conn.getEdges.side_effect = lambda v_type, v_id, edgeType="": (
        [{"to_type": "Entity", "to_id": "ent_1"}]
        if v_type == "Chunk"
        else [{"to_type": "Event", "to_id": "EV99"}]
    )
    expanded_res = retriever.retrieve("Event held at Sydney Olympic Park")
    assert expanded_res.is_graph_expanded
    assert len(expanded_res.evidence_chunks) > 0
    for ec in expanded_res.evidence_chunks:
        assert len(ec.provenance) > 0
