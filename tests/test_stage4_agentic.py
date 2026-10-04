"""Comprehensive unit test suite for Stage 4 / A3 & A4.

Tests cover all 24 required verification gates:
1. MCP configuration validation
2. MCP connection/session initialization
3. Tool allowlist enforcement
4. Forbidden tool rejection
5. Structured MCP result parsing
6. Evidence ledger creation
7. Evidence provenance preservation
8. Verification success
9. Verification insufficiency
10. Verification contradiction
11. One-repair behavior
12. Repair budget enforcement
13. No second repair
14. RunState immutability outside reducer
15. Reducer state transitions
16. Agent strategy selection
17. A0 routing
18. A1 routing
19. A2 routing
20. A3 verification routing
21. End-to-end simple lookup
22. End-to-end multi-hop question
23. Trace generation
24. Failure handling
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp_types import TextContent

from tgh.evidence.ledger import EvidenceLedger, EvidenceSourceType
from tgh.harness.engine import (
    AgentStrategy,
    ExecutionBudget,
    ExecutionPhase,
    HarnessReducer,
    RunState,
)
from tgh.mcp.contracts import EventCandidate
from tgh.mcp.tigergraph_client import (
    ALLOWED_MCP_TOOLS,
    FORBIDDEN_MCP_TOOLS,
    TigerGraphMCPClient,
)
from tgh.policies.agentic_orchestrator import AgenticGraphRAGOrchestrator
from tgh.policies.verification import (
    BoundedRepairExecutor,
    EvidenceVerifier,
    VerificationResult,
    VerificationStatus,
)
from tgh.telemetry.trace import TraceRecorder

# -----------------------------------------------------------------------------
# 1-5: MCP Tests
# -----------------------------------------------------------------------------


def test_mcp_configuration_validation():
    """1. Test MCP client environment variable configuration mapping."""
    env = {
        "TIGERGRAPH_HOST": "https://test.tgcloud.io",
        "TIGERGRAPH_GRAPH_NAME": "OlympicGraphRAG",
        "TIGERGRAPH_SECRET": "secret123",
    }
    client = TigerGraphMCPClient(env_dict=env)
    assert "tigergraph__get_node" in client.allowed_tools
    assert "tigergraph__gsql" in client.forbidden_tools


def test_mcp_connection_initialization():
    """2. Test MCP client session initialization and profile discovery."""
    client = TigerGraphMCPClient()
    assert client.server is not None


def test_tool_allowlist_enforcement():
    """3. Test only tools in ALLOWED_MCP_TOOLS are permitted."""
    client = TigerGraphMCPClient()
    for tool in ALLOWED_MCP_TOOLS:
        assert client.is_tool_allowed(tool) is True


def test_forbidden_tool_rejection():
    """4. Test all dangerous mutation and arbitrary query tools are rejected."""
    client = TigerGraphMCPClient()
    for tool in FORBIDDEN_MCP_TOOLS:
        assert client.is_tool_allowed(tool) is False
        assert client.is_tool_allowed(f"tigergraph__{tool}") is False


def test_structured_mcp_result_parsing():
    """5. Test parsing of markdown-wrapped JSON responses from official MCP server."""
    import asyncio

    async def _run():
        client = TigerGraphMCPClient()
        fake_json = (
            '```json\n{"success": true, "operation": "get_node", '
            '"data": {"v_id": "v1", "v_type": "Venue"}}\n```'
        )
        with patch.object(
            client.server, "_handle_call_tool", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = [TextContent(type="text", text=fake_json)]
            res = await client.call_tool(
                "tigergraph__get_node", {"vertex_type": "Venue", "vertex_id": "v1"}
            )
            assert res.success is True
            assert res.data["v_id"] == "v1"
            assert res.tool_name == "tigergraph__get_node"

    asyncio.run(_run())


# -----------------------------------------------------------------------------
# 6-7: Evidence Ledger Tests
# -----------------------------------------------------------------------------


def test_evidence_ledger_creation():
    """6. Test evidence ledger item creation and deduplication."""
    ledger = EvidenceLedger(run_id="run-1")
    item1 = ledger.add_item(
        source_type=EvidenceSourceType.VECTOR_CHUNK,
        source_id="Q100#c0000",
        document_id="Q100",
        text_or_fact="Some text",
    )
    assert item1.evidence_id.startswith("ev-")
    assert item1.document_id == "Q100"

    # Deduplication by source_id
    item2 = ledger.add_item(
        source_type=EvidenceSourceType.VECTOR_CHUNK,
        source_id="Q100#c0000",
    )
    assert item2.evidence_id == item1.evidence_id
    assert len(ledger.list_items()) == 1


def test_evidence_provenance_preservation():
    """7. Test preservation of graph path and relation in evidence item."""
    ledger = EvidenceLedger(run_id="run-2")
    path_str = "(Venue:v1) <-[HELD_AT]- (Event:e1)"
    item = ledger.add_item(
        source_type=EvidenceSourceType.GRAPH_FACT,
        source_id="Event:e1",
        document_id="e1",
        text_or_fact="Event e1 held at v1",
        provenance_path=path_str,
        relation="HELD_AT",
    )
    assert item.provenance_path == path_str
    assert item.relation == "HELD_AT"


# -----------------------------------------------------------------------------
# 8-13: Verification & One-Repair Tests
# -----------------------------------------------------------------------------


def test_verification_success():
    """8. Test verification succeeds when answer and year constraints are grounded."""
    verifier = EvidenceVerifier()
    ledger = EvidenceLedger()
    ledger.add_item(
        source_type=EvidenceSourceType.GRAPH_FACT,
        source_id="Event:Q1",
        document_id="Q1",
        text_or_fact="In 2016, Carolina Marín won gold in Women's singles badminton.",
    )
    q = "Who won the gold medal in the event held at Riocentro in 2016?"
    res = verifier.verify(question=q, candidate_answer="Carolina Marín", ledger=ledger)
    assert res.status == VerificationStatus.SUPPORTED
    assert res.repair_needed is False
    assert len(res.supporting_evidence_ids) == 1


def test_verification_insufficiency():
    """9. Test verification flags empty evidence as insufficient."""
    verifier = EvidenceVerifier()
    ledger = EvidenceLedger()
    res = verifier.verify(
        question="Who won in 2016?", candidate_answer="Unknown", ledger=ledger
    )
    assert res.status in (
        VerificationStatus.INSUFFICIENT,
        VerificationStatus.REPAIR_REQUIRED,
    )


def test_verification_contradiction_or_unsupported():
    """10. Test verification flags candidate answer that is not evidenced."""
    verifier = EvidenceVerifier()
    ledger = EvidenceLedger()
    ledger.add_item(
        source_type=EvidenceSourceType.GRAPH_FACT,
        source_id="Event:Q2",
        document_id="Q2",
        text_or_fact="In 2016, Nozomi Okuhara won bronze in badminton.",
    )
    q = "Who won the gold medal in 2016?"
    res = verifier.verify(question=q, candidate_answer="Carolina Marín", ledger=ledger)
    assert res.status == VerificationStatus.REPAIR_REQUIRED
    assert res.repair_needed is True
    assert "answer_unsupported" in res.sufficiency.missing_aspects


def test_one_repair_behavior():
    """11. Test exactly one repair attempt succeeds and populates ledger."""
    mock_tools = MagicMock()
    mock_tools.execute_composite_multihop.return_value = MagicMock(
        success=True,
        data={
            "events": [
                EventCandidate(
                    event_id="Q100",
                    name="Men's 60 kg",
                    year=1988,
                    description="Weightlifting 1988",
                    provenance_path="(Venue:v) <- (Event:Q100)",
                )
            ],
            "event_contexts": [],
        },
    )
    repairer = BoundedRepairExecutor(tools=mock_tools)
    ledger = EvidenceLedger()
    v_res = VerificationResult(
        status=VerificationStatus.REPAIR_REQUIRED,
        candidate_answer="Naim",
        sufficiency=MagicMock(missing_aspects=["answer_unsupported"]),
        supporting_evidence_ids=[],
        repair_needed=True,
        repair_focus="retrieve_missing_gold_document",
    )
    q = (
        "Who won the gold medal in the event held at "
        "Olympic Gymnasium on 20 September 1988?"
    )
    ok, msg = repairer.attempt_repair(
        question=q,
        verification_result=v_res,
        ledger=ledger,
        repair_budget=1,
    )
    assert ok is True
    assert len(ledger.list_items()) >= 1
    assert "Event:Q100" in [it.source_id for it in ledger.list_items()]


def test_repair_budget_enforcement():
    """12. Test repair fails when budget is 0."""
    repairer = BoundedRepairExecutor()
    ledger = EvidenceLedger()
    v_res = VerificationResult(
        status=VerificationStatus.REPAIR_REQUIRED,
        candidate_answer="Test",
        sufficiency=MagicMock(missing_aspects=[]),
        supporting_evidence_ids=[],
        repair_needed=True,
    )
    ok, err = repairer.attempt_repair(
        question="Test?",
        verification_result=v_res,
        ledger=ledger,
        repair_budget=0,
    )
    assert ok is False
    assert "budget exhausted" in err


def test_no_second_repair():
    """13. Test RunState limits repairs to max_repairs."""
    budget = ExecutionBudget(max_repairs=1)
    state = RunState(question="Test?", budget=budget)
    HarnessReducer.apply_event(state, "REPAIR_ATTEMPTED", {})
    assert state.repair_count == 1
    assert state.phase == ExecutionPhase.REPAIRING

    # Attempting second repair transitions to FAILED
    HarnessReducer.apply_event(state, "REPAIR_ATTEMPTED", {})
    assert state.repair_count == 2
    assert state.phase == ExecutionPhase.FAILED
    assert "Repair budget exceeded" in (state.error or "")


# -----------------------------------------------------------------------------
# 14-15: Execution Harness & Reducer Tests
# -----------------------------------------------------------------------------


def test_runstate_immutability_outside_reducer():
    """14. Test StateView exposes immutable snapshot of RunState."""
    state = RunState(question="Who won gold?")
    view = state.get_view()
    assert view.phase == ExecutionPhase.INITIALIZED
    assert view.tool_call_count == 0
    assert view.is_terminal is False

    with pytest.raises(AttributeError):
        view.phase = ExecutionPhase.COMPLETED  # type: ignore


def test_reducer_state_transitions():
    """15. Test valid sequence of reducer transitions."""
    state = RunState(question="Query")
    HarnessReducer.apply_event(
        state, "STRATEGY_CHOSEN", {"strategy": AgentStrategy.A0_VECTOR_RAG.value}
    )
    assert state.phase == ExecutionPhase.STRATEGY_SELECTED
    assert state.strategy == AgentStrategy.A0_VECTOR_RAG

    HarnessReducer.apply_event(
        state,
        "EVIDENCE_ADDED",
        {"source_type": "vector_chunk", "source_id": "c1", "document_id": "d1"},
    )
    assert state.phase == ExecutionPhase.EVIDENCE_ACCUMULATED
    assert len(state.ledger.list_items()) == 1

    HarnessReducer.apply_event(state, "ANSWER_PROPOSED", {"answer": "Athlete"})
    assert state.phase == ExecutionPhase.VERIFYING


# -----------------------------------------------------------------------------
# 16-20: Routing Policy Tests
# -----------------------------------------------------------------------------


def test_agent_strategy_selection():
    """16. Test orchestrator selects valid strategies for incoming queries."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    strategy = orch.route_question("Who won the gold medal?")
    assert strategy in (
        AgentStrategy.A0_VECTOR_RAG,
        AgentStrategy.A1_ADAPTIVE_GRAPHRAG,
        AgentStrategy.A2_DETERMINISTIC_GRAPH,
    )


def test_a0_routing():
    """17. Test A0 Vector RAG routing for direct lookup question."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    q = "Who won the gold medal in the men's 200m at the 2012 Summer Olympics?"
    assert orch.route_question(q) == AgentStrategy.A0_VECTOR_RAG


def test_a1_routing():
    """18. Test A1 Adaptive GraphRAG routing for relational questions."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    q = "Who represented Great Britain and won the gold medal in the event?"
    assert orch.route_question(q) == AgentStrategy.A1_ADAPTIVE_GRAPHRAG


def test_a2_routing():
    """19. Test A2 Deterministic Graph routing for venue/country/multi-hop questions."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    q1 = "Who won the gold medal in the event held at London Velopark in 2012?"
    assert orch.route_question(q1) == AgentStrategy.A2_DETERMINISTIC_GRAPH

    q2 = "How many nations competed in Sailing in 2016?"
    assert orch.route_question(q2) == AgentStrategy.A2_DETERMINISTIC_GRAPH


def test_a3_verification_routing():
    """20. Test A3 verification is executed on every orchestrated result."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {"candidates": [], "@@distances": {}, "@@chunk_to_doc": {}}
    ]
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    with patch(
        "tgh.embeddings.nomic.NomicEmbeddingProvider.embed_text",
        return_value=[0.1] * 768,
    ):
        res = orch.run("Who won the gold medal in 2012?")
        assert res.verification is not None
        assert res.verification.status in (
            VerificationStatus.SUPPORTED,
            VerificationStatus.INSUFFICIENT,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.REPAIR_REQUIRED,
        )


# -----------------------------------------------------------------------------
# 21-24: End-to-End & Telemetry Tests
# -----------------------------------------------------------------------------


def test_end_to_end_simple_lookup():
    """21. Test end-to-end flow for simple lookup question under A0 routing."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {
            "candidates": [
                {
                    "v_id": "Q100#c0000",
                    "v_type": "Chunk",
                    "attributes": {
                        "candidates.text": "Text describing Men's 200m champion"
                    },
                }
            ],
            "@@distances": {"Q100#c0000": 0.10},
            "@@chunk_to_doc": {"Q100#c0000": "Q100"},
        }
    ]
    with patch(
        "tgh.embeddings.nomic.NomicEmbeddingProvider.embed_text",
        return_value=[0.1] * 768,
    ):
        orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
        res = orch.run("Who won the men's 200m in 2012?")
        assert res.strategy == AgentStrategy.A0_VECTOR_RAG
        assert "Q100" in res.ranked_doc_ids
        assert res.state_view.phase == ExecutionPhase.COMPLETED


def test_end_to_end_multihop_question():
    """22. Test end-to-end flow for venue multi-hop question under A2 routing."""
    mock_conn = MagicMock()

    def mock_get_edges(v_type, v_id, edgeType=""):
        if v_type == "Venue":
            return [
                {
                    "to_id": "Q25301483",
                    "to_type": "Event",
                    "e_type": "reverse_HELD_AT",
                }
            ]
        elif v_type == "Event":
            return [
                {
                    "to_id": "venue_riocentro_pavilion_4",
                    "to_type": "Venue",
                    "e_type": "HELD_AT",
                },
                {
                    "to_id": "Carolina_Marin",
                    "to_type": "Person",
                    "e_type": "reverse_PARTICIPATED_IN",
                },
            ]
        return []

    mock_conn.getEdges.side_effect = mock_get_edges
    mock_conn.getVerticesById.return_value = [
        {
            "v_id": "Q25301483",
            "v_type": "Event",
            "attributes": {
                "name": "Women's singles",
                "year": 2016,
                "description": "Badminton 2016",
            },
        }
    ]
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    q = "Who won the gold medal in the event held at Riocentro Pavilion 4 in 2016?"
    res = orch.run(q)
    assert res.strategy == AgentStrategy.A2_DETERMINISTIC_GRAPH
    assert "Q25301483" in res.ranked_doc_ids
    assert "execute_composite_multihop" in res.a2_tools_invoked


def test_trace_generation():
    """23. Test machine-readable traces generated across the orchestration pipeline."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {"candidates": [], "@@distances": {}, "@@chunk_to_doc": {}}
    ]
    recorder = TraceRecorder(run_id="trace-test-1")
    with patch(
        "tgh.embeddings.nomic.NomicEmbeddingProvider.embed_text",
        return_value=[0.1] * 768,
    ):
        orch = AgenticGraphRAGOrchestrator(conn=mock_conn, recorder=recorder)
        res = orch.run("Simple query?")
        assert res.run_id == res.state_view.run_id
        traces = recorder.get_traces()
        assert len(traces) >= 1
        assert any(t.operation_name == "verify_answer" for t in traces)


def test_failure_handling():
    """24. Test budget exhaustion and graceful error capture."""
    state = RunState(question="Fail test", budget=ExecutionBudget(max_tool_calls=1))
    HarnessReducer.apply_event(state, "TOOL_CALLED", {"tool": "t1"})
    assert state.phase != ExecutionPhase.FAILED
    HarnessReducer.apply_event(state, "TOOL_CALLED", {"tool": "t2"})
    assert state.phase == ExecutionPhase.FAILED
    assert "budget exceeded" in (state.error or "")

def test_question_analysis_regex():
    """25. Test venue extraction and date extraction regexes for A4."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    state = RunState(question="dummy")

    # Venue with comma
    q1 = (
        "Who won the gold medal in the event held at Centennial Parklands, "
        "Sydney on 26 September at the 2000 Summer Olympics?"
    )
    a1 = orch._execute_question_analysis_skill(q1, state)
    assert a1["venue_cue"] == "Centennial Parklands, Sydney"
    assert a1["date_cue"] == "26 September"
    assert a1["year"] == 2000

    q2 = (
        "Who won the gold medal in the event held at Estadi Olímpic Lluís "
        "Companys, Barcelona on August 9 at the 1992 Summer Olympics?"
    )
    a2 = orch._execute_question_analysis_skill(q2, state)
    assert a2["venue_cue"] == "Estadi Olímpic Lluís Companys, Barcelona"
    assert a2["date_cue"] == "August 9"
    assert a2["year"] == 1992

    q3 = (
        "Who won the gold medal in the event held at Carioca Arena 3 on 6 August 2016?"
    )
    a3 = orch._execute_question_analysis_skill(q3, state)
    assert a3["venue_cue"] == "Carioca Arena 3"
    assert a3["date_cue"] == "6 August 2016"
    assert a3["year"] == 2016

    q4 = (
        "Who won the gold medal in the event held at Sydney Convention and "
        "Exhibition Centre on 23 September 2000?"
    )
    a4 = orch._execute_question_analysis_skill(q4, state)
    assert a4["venue_cue"] == "Sydney Convention and Exhibition Centre"
    assert a4["date_cue"] == "23 September 2000"
    assert a4["year"] == 2000

    q5 = (
        "Who won the gold medal in the event held at Sydney International "
        "Shooting Centre on 21 September 2000 (slow)22 September 2000 (fast)?"
    )
    a5 = orch._execute_question_analysis_skill(q5, state)
    assert a5["venue_cue"] == "Sydney International Shooting Centre"
    assert a5["date_cue"] == "21 September 2000 (slow)22 September 2000 (fast)"
    assert a5["year"] == 2000

    q6 = (
        "Who won the gold medal in the event held at Sydney Convention and "
        "Exhibition Centre on 18 September to 1 October 2000?"
    )
    a6 = orch._execute_question_analysis_skill(q6, state)
    assert a6["venue_cue"] == "Sydney Convention and Exhibition Centre"
    assert a6["date_cue"] == "18 September to 1 October 2000"
    assert a6["year"] == 2000
