"""Focused unit tests for Phase 8: Model-Driven DeepAgents A4 Architecture.

Tests cover all 11 required validation gates:
1. A4 actually invokes the model-driven DeepAgents agent.
2. A model action is passed through the Execution Harness.
3. Unsupported actions are rejected safely.
4. RunState remains harness-authoritative.
5. MCP allowlist remains enforced.
6. A2 tools remain deterministic.
7. Evidence Ledger remains authoritative.
8. Max repair budget remains 1 (no second repair permitted).
9. Model usage telemetry is recorded when available.
10. Deterministic candidate_answer remains available.
11. Existing A0/A1/A2 fallback behavior remains unchanged.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from tgh.harness.engine import (
    AgentStrategy,
    ExecutionBudget,
    ExecutionPhase,
    HarnessReducer,
    RunState,
    StateView,
)
from tgh.mcp.tigergraph_client import FORBIDDEN_MCP_TOOLS, TigerGraphMCPClient
from tgh.policies.agentic_orchestrator import (
    AgenticGraphRAGOrchestrator,
    AgenticOrchestrationResult,
)


class MockModel(BaseChatModel):
    """Deterministic mock chat model implementing bind_tools."""

    responses: list[AIMessage] = []
    idx: int = 0
    model_name: str = "mock-deepagents-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            resp = AIMessage(
                content="Mock default response",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )
        else:
            resp = self.responses[self.idx % len(self.responses)]
            self.idx += 1
        return ChatResult(generations=[ChatGeneration(message=resp)])

    @property
    def _llm_type(self) -> str:
        return "mock-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def test_1_a4_invokes_model_driven_deepagents_agent():
    """1. Test that A4 actually creates and invokes the DeepAgents agent."""
    mock_conn = MagicMock()
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "investigate_and_retrieve",
                "args": {
                    "strategy": "A0_Vector_RAG",
                    "question": "Who won gold?",
                },
                "id": "call_1",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 25, "output_tokens": 12, "total_tokens": 37},
    )
    final_msg = AIMessage(
        content="Winner is Person A",
        usage_metadata={"input_tokens": 35, "output_tokens": 8, "total_tokens": 43},
    )
    mock_llm = MockModel(responses=[tool_call_msg, final_msg])

    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, model=mock_llm)
    assert orch.deep_agent is not None

    with patch(
        "tgh.retrieval.vector.TigerGraphVectorRetriever.retrieve_seeds",
        return_value=[],
    ):
        res = orch.run("Who won gold?")
        assert res.model_name == "mock-deepagents-model"
        assert res.total_tokens == 80  # 37 + 43
        assert res.input_tokens == 60  # 25 + 35
        assert res.output_tokens == 20  # 12 + 8
        assert res.model_latency_ms >= 0.0


def test_2_model_action_passed_through_execution_harness():
    """2. Test that model tool call is intercepted and recorded by the harness."""
    mock_conn = MagicMock()
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "investigate_and_retrieve",
                "args": {
                    "strategy": "A2_Deterministic_Graph",
                    "question": "Event at Riocentro 2016",
                    "venue_cue": "Riocentro",
                    "year": 2016,
                },
                "id": "call_a2",
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 30, "output_tokens": 15, "total_tokens": 45},
    )
    final_msg = AIMessage(content="Grounded Answer")
    mock_llm = MockModel(responses=[tool_call_msg, final_msg])

    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, model=mock_llm)
    with patch.object(
        orch,
        "_execute_graph_reasoning_skill",
        return_value=(["doc_rio_1"], "Carolina"),
    ):
        res = orch.run("Event at Riocentro 2016")
        assert res.strategy == AgentStrategy.A2_DETERMINISTIC_GRAPH
        assert res.state_view.tool_call_count >= 1
        assert "question-analysis" in res.skills_invoked
        assert "retrieval-strategy-selection" in res.skills_invoked


def test_3_unsupported_actions_rejected():
    """3. Test that unsupported strategies or actions are rejected safely."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, enable_model_agent=False)

    state = RunState(question="test query")
    orch._active_run_state = state

    tools_dict = {t.name: t for t in orch._build_harness_tools()}
    inv_tool = tools_dict["investigate_and_retrieve"]

    # Invoking with invalid strategy
    res = inv_tool.invoke({
        "strategy": "INVALID_UNSUPPORTED_STRATEGY",
        "question": "test query",
    })
    assert "error" in res
    assert "not allowlisted" in res["error"]
    assert any(e.event_type == "TOOL_REJECTED" for e in state.events)


def test_4_runstate_remains_harness_authoritative():
    """4. Test that RunState cannot be modified directly and StateView is frozen."""
    state = RunState(question="What year?")
    view = state.get_view()
    assert isinstance(view, StateView)

    with pytest.raises(FrozenInstanceError):
        view.candidate_answer = "Unauthorized Modification"  # type: ignore[misc]

    # Only HarnessReducer transitions state
    HarnessReducer.apply_event(state, "ANSWER_PROPOSED", {"answer": "2016"})
    new_view = state.get_view()
    assert new_view.candidate_answer == "2016"


def test_5_mcp_allowlist_remains_enforced():
    """5. Test that all forbidden MCP tools are rejected by TigerGraphMCPClient."""
    client = TigerGraphMCPClient()
    for forbidden in FORBIDDEN_MCP_TOOLS:
        assert client.is_tool_allowed(forbidden) is False
        assert client.is_tool_allowed(f"tigergraph__{forbidden}") is False


def test_6_a2_tools_remain_deterministic():
    """6. Test that Layer 3 A2 tools return deterministic results with provenance."""
    mock_conn = MagicMock()
    mock_conn.getEdges.return_value = []
    mock_conn.getVerticesById.return_value = []

    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, enable_model_agent=False)
    res = orch.graph_tools.execute_composite_multihop(
        venue_id="venue_riocentro", year=2016
    )
    assert res.success is True
    assert isinstance(res.provenance, list)


def test_7_evidence_ledger_remains_authoritative():
    """7. Test that facts and document IDs must be registered in the Evidence Ledger."""
    state = RunState(question="Medalist inquiry")
    item = state.ledger.add_item(
        source_type="graph_fact",
        source_id="Person:Carolina_Marin",
        document_id="doc_badminton_2016",
        text_or_fact="Carolina Marin won gold medal in 2016",
    )
    assert item.evidence_id.startswith("ev-")
    assert state.ledger.get_document_ids() == ["doc_badminton_2016"]
    assert len(state.ledger.list_items()) == 1


def test_8_max_repair_budget_remains_one():
    """8. Test that at most 1 repair is allowed and subsequent repairs are rejected."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, enable_model_agent=False)
    state = RunState(question="Repair test", budget=ExecutionBudget(max_repairs=1))
    orch._active_run_state = state

    tools_dict = {t.name: t for t in orch._build_harness_tools()}
    repair_tool = tools_dict["verify_and_repair"]

    # First verification & repair call
    with patch.object(
        orch.repairer,
        "attempt_repair",
        return_value=(True, "Repair succeeded"),
    ):
        repair_res = repair_tool.invoke({"candidate_answer": "Wrong Answer"})
        assert "status" in repair_res
        assert state.repair_count <= 1

        # Attempting second repair: verify budget check
        HarnessReducer.apply_event(state, "REPAIR_ATTEMPTED", {"reason": "Test"})
        assert state.repair_count == 2
        assert state.phase == ExecutionPhase.FAILED
        assert "Repair budget exceeded" in (state.error or "")


def test_9_model_usage_telemetry_recorded():
    """9. Test that input, output, and total token usage are properly recorded."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "verify_and_repair",
                "args": {"candidate_answer": "Candidate 1"},
                "id": "c1",
                "type": "tool_call",
            }
        ],
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        },
    )
    final_msg = AIMessage(
        content="Candidate 1",
        usage_metadata={
            "input_tokens": 150,
            "output_tokens": 20,
            "total_tokens": 170,
        },
    )
    mock_llm = MockModel(responses=[tool_call_msg, final_msg])

    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, model=mock_llm)
    res = orch.run("Who won?")

    assert res.input_tokens == 250
    assert res.output_tokens == 60
    assert res.total_tokens == 310
    d = res.to_dict()
    assert d["input_tokens"] == 250
    assert d["output_tokens"] == 60
    assert d["total_tokens"] == 310


def test_10_deterministic_benchmark_compatible_candidate_answer_available():
    """10. Test benchmark candidate_answer remains extractable from state."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, enable_model_agent=False)

    fake_seed = MagicMock(
        chunk_id="c1",
        doc_id="doc_exact_id",
        similarity=0.95,
        text="Sample text",
    )
    with patch.object(
        orch.vector_retriever,
        "retrieve_seeds",
        return_value=[fake_seed],
    ):
        res = orch.run("Direct lookup question")
        assert res.candidate_answer == "doc_exact_id"
        assert res.ranked_doc_ids[0] == "doc_exact_id"
        assert isinstance(res, AgenticOrchestrationResult)


def test_11_existing_a0_a1_a2_behavior_unchanged():
    """11. Test that baseline routing without a model is fully preserved."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, enable_model_agent=False)

    # Route question without active model
    strat_venue = orch.route_question("Who won the event held at Riocentro in 2016?")
    assert strat_venue == AgentStrategy.A2_DETERMINISTIC_GRAPH

    strat_rel = orch.route_question("Which athlete represented France in 2012?")
    assert strat_rel == AgentStrategy.A1_ADAPTIVE_GRAPHRAG

    strat_simple = orch.route_question("Olympic motto definition")
    assert strat_simple == AgentStrategy.A0_VECTOR_RAG


def test_12_bounded_actions_run_a0_a1_a2():
    """12. Test model calls to run_a0_vector_rag, run_a1_graph_rag, run_a2."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, enable_model_agent=False)
    state = RunState(question="Who won gold at Riocentro 2016?")
    orch._active_run_state = state

    tools_dict = {t.name: t for t in orch._build_harness_tools()}
    assert "run_a0_vector_rag" in tools_dict
    assert "run_a1_graph_rag" in tools_dict
    assert "run_a2_graph_reasoning" in tools_dict

    with patch.object(orch.vector_retriever, "retrieve_seeds", return_value=[]):
        res_a0 = tools_dict["run_a0_vector_rag"].invoke({"query": "Gold medal"})
        assert res_a0["strategy"] == AgentStrategy.A0_VECTOR_RAG.value

    with patch.object(
        orch.graphrag_retriever,
        "retrieve",
        return_value=MagicMock(ranked_doc_ids=["doc_1"], evidence_chunks=[]),
    ):
        res_a1 = tools_dict["run_a1_graph_rag"].invoke({"query": "Gold medal"})
        assert res_a1["strategy"] == AgentStrategy.A1_ADAPTIVE_GRAPHRAG.value

    with patch.object(
        orch,
        "_execute_graph_reasoning_skill",
        return_value=(["doc_a2"], "Person A2"),
    ):
        res_a2 = tools_dict["run_a2_graph_reasoning"].invoke({
            "venue_cue": "Riocentro",
            "year": 2016,
        })
        assert res_a2["strategy"] == AgentStrategy.A2_DETERMINISTIC_GRAPH.value


def test_13_second_repair_rejected_and_terminates_safely():
    """13. Test that a second repair is rejected and state terminates safely."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn, enable_model_agent=False)
    state = RunState(question="Repair check", budget=ExecutionBudget(max_repairs=1))
    orch._active_run_state = state

    tools_dict = {t.name: t for t in orch._build_harness_tools()}
    repair_tool = tools_dict["verify_and_repair"]

    with patch.object(
        orch.repairer,
        "attempt_repair",
        return_value=(True, "First repair OK"),
    ):
        # First repair succeeds
        res1 = repair_tool.invoke({"candidate_answer": "Answer 1"})
        assert res1["repair_count"] == 1

        # Second repair attempted: harness rejects it
        HarnessReducer.apply_event(
            state, "REPAIR_ATTEMPTED", {"reason": "Second repair attempt"}
        )
        assert state.repair_count == 2
        assert state.phase == ExecutionPhase.FAILED
        assert "Repair budget exceeded" in (state.error or "")


