"""Layer 4: Execution Harness authoritative RunState, StateView, Reducer, and Events."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from tgh.evidence.ledger import EvidenceLedger


class AgentStrategy(StrEnum):
    """Pipeline strategy selected for the current execution."""

    A0_VECTOR_RAG = "A0_Vector_RAG"
    A1_ADAPTIVE_GRAPHRAG = "A1_Adaptive_GraphRAG"
    A2_DETERMINISTIC_GRAPH = "A2_Deterministic_Graph"
    A3_VERIFICATION = "A3_Verification"
    A4_AGENTIC_ORCHESTRATION = "A4_Agentic_Orchestration"


class ExecutionPhase(StrEnum):
    """Lifecycle phase of the execution harness."""

    INITIALIZED = "INITIALIZED"
    STRATEGY_SELECTED = "STRATEGY_SELECTED"
    RETRIEVING = "RETRIEVING"
    EVIDENCE_ACCUMULATED = "EVIDENCE_ACCUMULATED"
    VERIFYING = "VERIFYING"
    REPAIRING = "REPAIRING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class HarnessEvent:
    """Immutable event record representing a state transition."""

    event_id: str
    event_type: str
    timestamp: float
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionBudget:
    """Safety and resource budgets for harness execution."""

    max_tool_calls: int = 15
    max_repairs: int = 1
    max_evidence_items: int = 50
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class StateView:
    """Read-only view of authoritative execution state exposed to agents and tools."""

    run_id: str
    question: str
    phase: ExecutionPhase
    strategy: AgentStrategy | None
    tool_call_count: int
    repair_count: int
    candidate_answer: str | None
    ranked_doc_ids: list[str]
    evidence_count: int
    is_terminal: bool
    skills_invoked: tuple[str, ...] = ()


class RunState:
    """Authoritative execution state owned exclusively by the Layer 4 Harness.

    No external agent or tool is permitted to mutate RunState directly.
    All transitions must pass through the Harness Reducer.
    """

    def __init__(
        self,
        question: str,
        run_id: str | None = None,
        budget: ExecutionBudget | None = None,
    ) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self.question = question
        self.budget = budget or ExecutionBudget()
        self.phase = ExecutionPhase.INITIALIZED
        self.strategy: AgentStrategy | None = None
        self.tool_call_count = 0
        self.repair_count = 0
        self.candidate_answer: str | None = None
        self.ranked_doc_ids: list[str] = []
        self.skills_invoked: list[str] = []
        self.ledger = EvidenceLedger(run_id=self.run_id)
        self.events: list[HarnessEvent] = []
        self.error: str | None = None
        self.start_time = time.perf_counter()
        self.end_time: float | None = None

    def get_view(self) -> StateView:
        """Produce an immutable snapshot of current state."""
        return StateView(
            run_id=self.run_id,
            question=self.question,
            phase=self.phase,
            strategy=self.strategy,
            tool_call_count=self.tool_call_count,
            repair_count=self.repair_count,
            candidate_answer=self.candidate_answer,
            ranked_doc_ids=list(self.ranked_doc_ids),
            evidence_count=len(self.ledger.list_items()),
            is_terminal=self.phase in (ExecutionPhase.COMPLETED, ExecutionPhase.FAILED),
            skills_invoked=tuple(self.skills_invoked),
        )


class HarnessReducer:
    """Authoritative reducer governing all state transitions in RunState."""

    @staticmethod
    def apply_event(
        state: RunState, event_type: str, payload: dict[str, Any]
    ) -> RunState:
        """Transition RunState based on an explicit event."""
        event_id = f"evt-{len(state.events) + 1:04d}-{str(uuid.uuid4())[:6]}"
        now = time.perf_counter()
        ev = HarnessEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=now,
            payload=payload,
        )
        state.events.append(ev)

        if event_type == "SKILL_ACTIVATED":
            s_name = payload.get("skill", "")
            if s_name and s_name not in state.skills_invoked:
                state.skills_invoked.append(s_name)

        elif event_type == "SKILL_EXECUTED":
            pass

        elif event_type == "STRATEGY_CHOSEN":
            state.strategy = AgentStrategy(payload["strategy"])
            state.phase = ExecutionPhase.STRATEGY_SELECTED

        elif event_type == "TOOL_CALLED":
            state.tool_call_count += 1
            if state.tool_call_count > state.budget.max_tool_calls:
                state.phase = ExecutionPhase.FAILED
                state.error = (
                    f"Tool call budget exceeded: {state.tool_call_count} > "
                    f"{state.budget.max_tool_calls}"
                )

        elif event_type == "EVIDENCE_ADDED":
            state.ledger.add_item(
                source_type=payload["source_type"],
                source_id=payload["source_id"],
                document_id=payload.get("document_id"),
                text_or_fact=payload.get("text_or_fact", ""),
                provenance_path=payload.get("provenance_path"),
                relation=payload.get("relation"),
                confidence=payload.get("confidence", 1.0),
                metadata=payload.get("metadata"),
            )
            state.phase = ExecutionPhase.EVIDENCE_ACCUMULATED

        elif event_type == "DOCS_RANKED":
            state.ranked_doc_ids = payload["ranked_doc_ids"]

        elif event_type == "ANSWER_PROPOSED":
            state.candidate_answer = payload["answer"]
            state.phase = ExecutionPhase.VERIFYING

        elif event_type == "REPAIR_ATTEMPTED":
            state.repair_count += 1
            if state.repair_count > state.budget.max_repairs:
                state.phase = ExecutionPhase.FAILED
                state.error = (
                    f"Repair budget exceeded: {state.repair_count} > "
                    f"{state.budget.max_repairs}"
                )
            else:
                state.phase = ExecutionPhase.REPAIRING

        elif event_type == "EXECUTION_COMPLETED":
            state.phase = ExecutionPhase.COMPLETED
            state.end_time = now

        elif event_type == "EXECUTION_FAILED":
            state.phase = ExecutionPhase.FAILED
            state.error = payload.get("error", "Unknown error")
            state.end_time = now

        return state
