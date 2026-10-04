"""Focused tests for DeepAgents skills integration in A4 Agentic GraphRAG.

Verifies:
1. All 5 required skills discovered via deepagents.middleware.skills
2. Skill metadata compliance (Agent Skills specification)
3. Question Analysis skill execution
4. Retrieval Strategy Selection skill execution
5. Graph Reasoning skill execution with provenance
6. Evidence Verification skill execution with bounded repair
7. Answer Synthesis skill grounding in Evidence Ledger
8. Harness authoritativeness (RunState cannot be mutated directly)
9. Forbidden MCP tools remain strictly blocked
10. Hidden benchmark (eval_hidden.jsonl) is never accessed
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import _list_skills

from tgh.harness.engine import (
    AgentStrategy,
    ExecutionPhase,
    HarnessReducer,
    RunState,
    StateView,
)
from tgh.mcp.tigergraph_client import FORBIDDEN_MCP_TOOLS, TigerGraphMCPClient
from tgh.policies.agentic_orchestrator import AgenticGraphRAGOrchestrator
from tgh.policies.verification import VerificationStatus


def test_five_skills_registered_via_deepagents():
    """1. Test that all five required skills are discovered by DeepAgents."""
    repo_root = Path(__file__).resolve().parent.parent
    backend = FilesystemBackend(root_dir=repo_root)
    skills = _list_skills(backend, "/skills/")
    skill_names = {s["name"] for s in skills}

    expected = {
        "question-analysis",
        "retrieval-strategy-selection",
        "graph-reasoning",
        "evidence-verification",
        "answer-synthesis",
    }
    assert expected.issubset(skill_names), f"Missing skills: {expected - skill_names}"
    assert len(skills) == 5


def test_skill_metadata_compliance():
    """2. Test that each skill adheres to the Agent Skills specification."""
    repo_root = Path(__file__).resolve().parent.parent
    backend = FilesystemBackend(root_dir=repo_root)
    skills = _list_skills(backend, "/skills/")

    for s in skills:
        assert 1 <= len(s["name"]) <= 64
        assert s["name"] == s["name"].lower()
        assert not s["name"].startswith("-")
        assert not s["name"].endswith("-")
        assert "--" not in s["name"]
        assert 1 <= len(s["description"]) <= 1024
        assert s["path"].endswith("SKILL.md")


def test_orchestrator_loads_deepagents_skills():
    """3. Test that AgenticGraphRAGOrchestrator loads DeepAgents skills."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    assert len(orch.skills) == 5
    assert "question-analysis" in orch.skills
    assert "retrieval-strategy-selection" in orch.skills
    assert "graph-reasoning" in orch.skills
    assert "evidence-verification" in orch.skills
    assert "answer-synthesis" in orch.skills


def test_question_analysis_skill_execution():
    """4. Test Question Analysis skill extracts venue, year, and intent."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    state = RunState(question="test")

    q = (
        "Who won the gold medal in the event held at Riocentro – Pavilion 4 "
        "on 11–19 August at the 2016 Summer Olympics?"
    )
    analysis = orch._execute_question_analysis_skill(q, state)
    assert analysis["venue_cue"] == "Riocentro – Pavilion 4"
    assert analysis["year"] == 2016
    assert analysis["is_venue_multihop"] is True
    assert "question-analysis" in state.skills_invoked


def test_retrieval_strategy_selection_skill_execution():
    """5. Test Retrieval Strategy Selection skill routes correctly."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    state = RunState(question="test")

    a_venue = {
        "is_venue_multihop": True,
        "is_nation_agg": False,
        "is_relational": False,
    }
    strat1 = orch._execute_retrieval_strategy_selection_skill(a_venue, state)
    assert strat1 == AgentStrategy.A2_DETERMINISTIC_GRAPH

    a_rel = {
        "is_venue_multihop": False,
        "is_nation_agg": False,
        "is_relational": True,
    }
    strat2 = orch._execute_retrieval_strategy_selection_skill(a_rel, state)
    assert strat2 == AgentStrategy.A1_ADAPTIVE_GRAPHRAG

    a_simple = {
        "is_venue_multihop": False,
        "is_nation_agg": False,
        "is_relational": False,
    }
    strat3 = orch._execute_retrieval_strategy_selection_skill(a_simple, state)
    assert strat3 == AgentStrategy.A0_VECTOR_RAG
    assert "retrieval-strategy-selection" in state.skills_invoked


def test_graph_reasoning_skill_execution():
    """6. Test Graph Reasoning skill executes bounded traversal with provenance."""
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
                "description": "Badminton",
            },
        }
    ]

    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    state = RunState(question="test")
    analysis = {
        "venue_cue": "Riocentro Pavilion 4",
        "year": 2016,
    }
    a2_calls: list[str] = []
    ranked_docs, cand_ans = orch._execute_graph_reasoning_skill(
        "Who won gold at Riocentro Pavilion 4 in 2016?", analysis, state, a2_calls
    )
    assert "Q25301483" in ranked_docs
    assert "graph-reasoning" in state.skills_invoked
    assert "execute_composite_multihop" in a2_calls
    assert len(state.ledger.list_items()) > 0


def test_evidence_verification_and_repair_skill():
    """7. Test Evidence Verification skill triggers targeted repair within budget."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    state = RunState(question="Who won gold in 2016?")
    # Empty ledger forces REPAIR_REQUIRED
    verif, repair_used, ranked = orch._execute_evidence_verification_skill(
        "Who won gold in 2016?", "Unknown", state, []
    )
    assert "evidence-verification" in state.skills_invoked
    assert state.repair_count <= 1


def test_answer_synthesis_skill_grounding():
    """8. Test Answer Synthesis skill rejects ungrounded candidate answers."""
    mock_conn = MagicMock()
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    state = RunState(question="test")
    verif = MagicMock()
    verif.status = VerificationStatus.INSUFFICIENT
    final_ans = orch._execute_answer_synthesis_skill("Fabricated Answer", verif, state)
    assert final_ans == "INSUFFICIENT_EVIDENCE"
    assert "answer-synthesis" in state.skills_invoked


def test_harness_authoritativeness_with_skills():
    """9. Test that RunState is immutable outside the reducer and records skills."""
    state = RunState(question="Who won in 2016?")
    view1 = state.get_view()
    assert isinstance(view1, StateView)
    assert view1.skills_invoked == ()

    HarnessReducer.apply_event(state, "SKILL_ACTIVATED", {"skill": "question-analysis"})
    view2 = state.get_view()
    assert view2.skills_invoked == ("question-analysis",)

    # StateView is immutable
    with pytest.raises(FrozenInstanceError):
        view2.candidate_answer = "Mutated"  # type: ignore[misc]


def test_forbidden_mcp_tools_blocked():
    """10. Test that forbidden MCP tools remain strictly blocked from the agent."""
    client = TigerGraphMCPClient()
    for forbidden in FORBIDDEN_MCP_TOOLS:
        assert client.is_tool_allowed(forbidden) is False
        assert client.is_tool_allowed(f"tigergraph__{forbidden}") is False


def test_hidden_benchmark_never_accessed():
    """11. Test that eval_hidden.jsonl is never referenced or accessed in code."""
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "eval_hidden" not in content, f"eval_hidden referenced in {py_file}"


def test_end_to_end_orchestration_skills_execution():
    """12. Test full end-to-end run invokes skills and produces grounded output."""
    mock_conn = MagicMock()
    mock_conn.runInstalledQuery.return_value = [
        {"candidates": [], "@@distances": {}, "@@chunk_to_doc": {}}
    ]
    orch = AgenticGraphRAGOrchestrator(conn=mock_conn)
    with patch(
        "tgh.embeddings.nomic.NomicEmbeddingProvider.embed_text",
        return_value=[0.1] * 768,
    ):
        q = "Who won the gold medal in the men's 200m at the 2012 Summer Olympics?"
        res = orch.run(q)
        assert len(res.skills_invoked) >= 4
        assert "question-analysis" in res.skills_invoked
        assert "retrieval-strategy-selection" in res.skills_invoked
        assert "evidence-verification" in res.skills_invoked
        assert "answer-synthesis" in res.skills_invoked
        assert res.state_view.phase == ExecutionPhase.COMPLETED
