"""Layer 5: A4 Agentic GraphRAG Orchestrator integrating MCP and Skills."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.skills import (
    SkillMetadata,
    SkillsMiddleware,
    _list_skills,
)
from langchain_core.tools import tool

from tgh.embeddings.nomic import NomicEmbeddingProvider
from tgh.evidence.ledger import EvidenceItem
from tgh.harness.engine import (
    AgentStrategy,
    ExecutionBudget,
    ExecutionPhase,
    HarnessReducer,
    RunState,
    StateView,
)
from tgh.mcp.graph_tools import TigerGraphTools
from tgh.mcp.tigergraph_client import ALLOWED_MCP_TOOLS, TigerGraphMCPClient
from tgh.policies.verification import (
    BoundedRepairExecutor,
    EvidenceVerifier,
    VerificationResult,
    VerificationStatus,
)
from tgh.retrieval.graphrag import GraphRAGRetriever
from tgh.retrieval.vector import TigerGraphVectorRetriever
from tgh.telemetry.trace import TraceRecorder


@dataclass
class AgenticOrchestrationResult:
    """Final output of A4 Agentic GraphRAG reasoning."""

    run_id: str
    question: str
    strategy: AgentStrategy
    state_view: StateView
    ranked_doc_ids: list[str]
    candidate_answer: str
    verification: VerificationResult
    repair_used: bool
    latency_ms: float
    mcp_tools_invoked: list[str]
    a2_tools_invoked: list[str]
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    skills_invoked: list[str] = field(default_factory=list)
    model_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_latency_ms: float = 0.0
    model_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "question": self.question,
            "strategy": self.strategy.value,
            "ranked_doc_ids": self.ranked_doc_ids,
            "candidate_answer": self.candidate_answer,
            "verification": self.verification.to_dict(),
            "repair_used": self.repair_used,
            "latency_ms": self.latency_ms,
            "mcp_tools_invoked": self.mcp_tools_invoked,
            "a2_tools_invoked": self.a2_tools_invoked,
            "skills_invoked": self.skills_invoked,
            "evidence_count": self.state_view.evidence_count,
            "model_name": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_latency_ms": self.model_latency_ms,
            "model_answer": self.model_answer,
        }


logger = logging.getLogger(__name__)


class AgenticGraphRAGOrchestrator:
    """Layer 5 Orchestrator coordinating DeepAgents agent, skills, and harness."""

    def __init__(
        self,
        conn: Any,
        embedding_provider: NomicEmbeddingProvider | None = None,
        mcp_client: TigerGraphMCPClient | None = None,
        recorder: TraceRecorder | None = None,
        skills_dir: Path | None = None,
        model: Any | None = None,
        enable_model_agent: bool = True,
    ) -> None:
        self.conn = conn
        self.recorder = recorder or TraceRecorder()
        self.embedding_provider = embedding_provider or NomicEmbeddingProvider(
            dimension=768
        )
        self.vector_retriever = TigerGraphVectorRetriever(
            conn=conn,
            provider=self.embedding_provider,
            top_k=20,
        )
        self.graphrag_retriever = GraphRAGRetriever(
            conn=conn,
            embedding_provider=self.embedding_provider,
        )
        self.graph_tools = TigerGraphTools(
            conn=conn,
            recorder=self.recorder,
        )
        self.mcp_client = mcp_client
        self.verifier = EvidenceVerifier(recorder=self.recorder)
        self.repairer = BoundedRepairExecutor(
            tools=self.graph_tools,
            graphrag_retriever=self.graphrag_retriever,
            recorder=self.recorder,
        )

        # DeepAgents Skills System integration (Agent Skills specification)
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        self.skills_dir = skills_dir or (repo_root / "skills")
        self.backend = FilesystemBackend(root_dir=self.skills_dir.parent)
        self.skills_middleware = SkillsMiddleware(
            backend=self.backend,
            sources=[f"/{self.skills_dir.name}/"],
        )
        raw_skills = _list_skills(self.backend, f"/{self.skills_dir.name}/")
        self.skills: dict[str, SkillMetadata] = {s["name"]: s for s in raw_skills}

        # Model and DeepAgents Agent configuration
        self.model = model
        self.model_name: str | None = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or (type(model).__name__ if model is not None else None)
        )
        if self.model is None and enable_model_agent:
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
            if gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI

                    model_name = os.getenv("LLM_MODEL_NAME") or "gemini-3.5-flash-lite"
                    self.model = ChatGoogleGenerativeAI(
                        model=model_name,
                        api_key=gemini_key,
                        max_retries=1,
                        temperature=0.0,
                    )
                    self.model_name = model_name
                except Exception as e:
                    logger.warning(f"Could not initialize ChatGoogleGenerativeAI: {e}")
                    self.model = None

        # Build DeepAgents model-driven agent if model is configured
        self._active_run_state: RunState | None = None
        self._active_mcp_tools: list[str] | None = None
        self._active_a2_tools: list[str] | None = None
        self.deep_agent = self._create_deep_agent() if self.model is not None else None

    def _build_harness_tools(self) -> list[Any]:
        """Construct allowlisted tools bound to the authoritative execution harness."""

        def _execute_retrieval_action(
            strat_enum: AgentStrategy,
            question: str,
            venue_cue: str = "",
            year: int = 0,
            date_cue: str = "",
            tool_name: str = "investigate_and_retrieve",
        ) -> dict[str, Any]:
            state = self._active_run_state
            if state is None:
                return {"error": "No active harness run state"}

            # Validate tool call budget
            if state.tool_call_count >= state.budget.max_tool_calls:
                HarnessReducer.apply_event(
                    state,
                    "TOOL_REJECTED",
                    {"tool": tool_name, "reason": "Budget exceeded"},
                )
                return {"error": "Tool call budget exceeded"}

            HarnessReducer.apply_event(state, "TOOL_CALLED", {"tool": tool_name})
            HarnessReducer.apply_event(
                state, "STRATEGY_CHOSEN", {"strategy": strat_enum.value}
            )

            # Record skills activated & executed
            HarnessReducer.apply_event(
                state, "SKILL_ACTIVATED", {"skill": "question-analysis"}
            )
            analysis = {
                "venue_cue": venue_cue or None,
                "date_cue": date_cue or None,
                "year": year or None,
                "is_nation_agg": "how many nations" in question.lower(),
                "is_relational": strat_enum == AgentStrategy.A1_ADAPTIVE_GRAPHRAG,
                "is_venue_multihop": bool(venue_cue)
                or strat_enum == AgentStrategy.A2_DETERMINISTIC_GRAPH,
            }
            HarnessReducer.apply_event(
                state,
                "SKILL_EXECUTED",
                {"skill": "question-analysis", "analysis": analysis},
            )
            HarnessReducer.apply_event(
                state, "SKILL_ACTIVATED", {"skill": "retrieval-strategy-selection"}
            )
            HarnessReducer.apply_event(
                state,
                "SKILL_EXECUTED",
                {"skill": "retrieval-strategy-selection", "selected": strat_enum.value},
            )

            ranked_docs: list[str] = []
            candidate_answer = ""

            if strat_enum == AgentStrategy.A2_DETERMINISTIC_GRAPH:
                a2_tools = (
                    self._active_a2_tools
                    if self._active_a2_tools is not None
                    else []
                )
                ranked_docs, candidate_answer = (
                    self._execute_graph_reasoning_skill(
                        question, analysis, state, a2_tools
                    )
                )
            elif strat_enum == AgentStrategy.A1_ADAPTIVE_GRAPHRAG:
                rag_res = self.graphrag_retriever.retrieve(question, top_k_seeds=20)
                ranked_docs = rag_res.ranked_doc_ids
                for ec in rag_res.evidence_chunks[:10]:
                    HarnessReducer.apply_event(
                        state,
                        "EVIDENCE_ADDED",
                        {
                            "source_type": (
                                "vector_chunk"
                                if ec.source == "seed"
                                else "graph_fact"
                            ),
                            "source_id": ec.chunk_id,
                            "document_id": ec.doc_id,
                            "text_or_fact": (
                                f"Evidence for {ec.doc_id} score={ec.score:.4f}"
                            ),
                            "provenance_path": (
                                ec.provenance[0] if ec.provenance else None
                            ),
                            "confidence": ec.score,
                        },
                    )
                if rag_res.evidence_chunks:
                    candidate_answer = rag_res.evidence_chunks[0].doc_id
            else:
                # A0 Vector RAG
                seeds = self.vector_retriever.retrieve_seeds(question, top_k=20)
                ranked_docs = [s.doc_id for s in seeds if s.doc_id]
                for s in seeds[:10]:
                    HarnessReducer.apply_event(
                        state,
                        "EVIDENCE_ADDED",
                        {
                            "source_type": "vector_chunk",
                            "source_id": s.chunk_id,
                            "document_id": s.doc_id,
                            "text_or_fact": s.text or f"Chunk {s.chunk_id}",
                            "confidence": s.similarity,
                        },
                    )
                if ranked_docs:
                    candidate_answer = ranked_docs[0]

            seen: set[str] = set()
            unique_ranked: list[str] = []
            for d in ranked_docs:
                if d and d not in seen:
                    seen.add(d)
                    unique_ranked.append(d)

            HarnessReducer.apply_event(
                state, "DOCS_RANKED", {"ranked_doc_ids": unique_ranked}
            )
            HarnessReducer.apply_event(
                state, "ANSWER_PROPOSED", {"answer": candidate_answer}
            )

            return {
                "strategy": strat_enum.value,
                "retrieved_count": len(unique_ranked),
                "ranked_doc_ids": unique_ranked[:5],
                "candidate_answer": candidate_answer,
                "evidence_count": len(state.ledger.list_items()),
            }

        @tool
        def run_a0_vector_rag(query: str = "") -> dict[str, Any]:
            """Execute bounded A0 Vector RAG retrieval through Execution Harness."""
            q = query or (
                self._active_run_state.question if self._active_run_state else ""
            )
            return _execute_retrieval_action(
                AgentStrategy.A0_VECTOR_RAG,
                question=q,
                tool_name="run_a0_vector_rag",
            )

        @tool
        def run_a1_graph_rag(query: str = "") -> dict[str, Any]:
            """Execute bounded A1 Adaptive GraphRAG through Execution Harness."""
            q = query or (
                self._active_run_state.question if self._active_run_state else ""
            )
            return _execute_retrieval_action(
                AgentStrategy.A1_ADAPTIVE_GRAPHRAG,
                question=q,
                tool_name="run_a1_graph_rag",
            )

        @tool
        def run_a2_graph_reasoning(
            venue_cue: str = "",
            year: int = 0,
            date_cue: str = "",
        ) -> dict[str, Any]:
            """Execute bounded A2 Deterministic Graph Reasoning through Harness."""
            q = self._active_run_state.question if self._active_run_state else ""
            return _execute_retrieval_action(
                AgentStrategy.A2_DETERMINISTIC_GRAPH,
                question=q,
                venue_cue=venue_cue,
                year=year,
                date_cue=date_cue,
                tool_name="run_a2_graph_reasoning",
            )

        @tool
        def investigate_and_retrieve(
            strategy: str,
            question: str,
            venue_cue: str = "",
            year: int = 0,
            date_cue: str = "",
        ) -> dict[str, Any]:
            """Investigate question and retrieve evidence using chosen strategy.

            Args:
                strategy: Strategy name: 'A0_Vector_RAG', 'A1_Adaptive_GraphRAG',
                    or 'A2_Deterministic_Graph'.
                question: The natural language question to investigate.
                venue_cue: Optional venue name if detected.
                year: Optional Olympic year if detected (e.g. 2016).
                date_cue: Optional date constraint (e.g. '11–19 August').
            """
            state = self._active_run_state
            if state is None:
                return {"error": "No active harness run state"}

            valid_strategies = {s.value for s in AgentStrategy}
            if strategy not in valid_strategies:
                HarnessReducer.apply_event(
                    state,
                    "TOOL_REJECTED",
                    {"tool": "investigate_and_retrieve", "strategy": strategy},
                )
                return {
                    "error": (
                        f"Strategy {strategy} is not allowlisted. "
                        f"Must be one of: {list(valid_strategies)}"
                    )
                }

            return _execute_retrieval_action(
                AgentStrategy(strategy),
                question=question,
                venue_cue=venue_cue,
                year=year,
                date_cue=date_cue,
                tool_name="investigate_and_retrieve",
            )

        @tool
        def verify_and_repair(candidate_answer: str = "") -> dict[str, Any]:
            """Verify candidate answer against Evidence Ledger and repair (max 1)."""
            state = self._active_run_state
            if state is None:
                return {"error": "No active harness run state"}

            if state.tool_call_count >= state.budget.max_tool_calls:
                HarnessReducer.apply_event(
                    state,
                    "TOOL_REJECTED",
                    {"tool": "verify_and_repair", "reason": "Budget exceeded"},
                )
                return {"error": "Tool call budget exceeded"}

            HarnessReducer.apply_event(
                state, "TOOL_CALLED", {"tool": "verify_and_repair"}
            )

            cand = candidate_answer or (state.candidate_answer or "")
            verification, repair_used, unique_ranked = (
                self._execute_evidence_verification_skill(
                    state.question, cand, state, state.ranked_doc_ids
                )
            )

            return {
                "status": verification.status.value,
                "repair_used": repair_used,
                "repair_count": state.repair_count,
                "ranked_doc_ids": unique_ranked[:5],
                "candidate_answer": cand,
            }

        @tool
        def mcp_get_node(vertex_type: str, vertex_id: str) -> dict[str, Any]:
            """Official TigerGraph MCP tool: Read node attributes (allowlisted only)."""
            state = self._active_run_state
            if state is None:
                return {"error": "No active harness run state"}

            if "tigergraph__get_node" not in ALLOWED_MCP_TOOLS:
                HarnessReducer.apply_event(
                    state, "TOOL_REJECTED", {"tool": "tigergraph__get_node"}
                )
                return {"error": "Forbidden MCP tool"}

            HarnessReducer.apply_event(
                state, "TOOL_CALLED", {"tool": "tigergraph__get_node"}
            )
            if self._active_mcp_tools is not None:
                self._active_mcp_tools.append("tigergraph__get_node")

            res = self.graph_tools.get_node(vertex_type, vertex_id)
            return res.to_dict()

        @tool
        def mcp_get_edges(
            source_type: str, source_id: str, edge_type: str = ""
        ) -> dict[str, Any]:
            """Official TigerGraph MCP tool: Read vertex edges (allowlisted only)."""
            state = self._active_run_state
            if state is None:
                return {"error": "No active harness run state"}

            if "tigergraph__get_edges" not in ALLOWED_MCP_TOOLS:
                HarnessReducer.apply_event(
                    state, "TOOL_REJECTED", {"tool": "tigergraph__get_edges"}
                )
                return {"error": "Forbidden MCP tool"}

            HarnessReducer.apply_event(
                state, "TOOL_CALLED", {"tool": "tigergraph__get_edges"}
            )
            if self._active_mcp_tools is not None:
                self._active_mcp_tools.append("tigergraph__get_edges")

            res = self.graph_tools.get_edges(
                source_type, source_id, edge_type=edge_type
            )
            return res.to_dict()

        return [
            run_a0_vector_rag,
            run_a1_graph_rag,
            run_a2_graph_reasoning,
            investigate_and_retrieve,
            verify_and_repair,
            mcp_get_node,
            mcp_get_edges,
        ]

    def _create_deep_agent(self) -> Any:
        """Create genuine DeepAgents agent with skills and allowlisted tools."""
        system_prompt = (
            "You are the Olympic GraphRAG Agent governed by an authoritative "
            "Execution Harness.\n"
            "Your decisions must strictly follow the procedural guidance in /skills/:\n"
            "- question-analysis: Analyze entity mentions, venues, temporal "
            "constraints, and intent.\n"
            "- retrieval-strategy-selection:\n"
            "    * If the question has venue/location or aggregation (e.g. held at "
            "<venue>, how many nations), call run_a2_graph_reasoning.\n"
            "    * If the question asks about athlete representation or "
            "participation, call run_a1_graph_rag.\n"
            "    * Otherwise, call run_a0_vector_rag.\n"
            "- graph-reasoning: Bounded graph traversal for candidate event "
            "filtering.\n"
            "- evidence-verification: Verify candidate answers against the Evidence "
            "Ledger (max 1 repair permitted).\n"
            "- answer-synthesis: Ground answers strictly in verified evidence.\n\n"
            "Workflow:\n"
            "1. Analyze question and call appropriate bounded retrieval action:\n"
            "   - run_a2_graph_reasoning for venue/multihop queries\n"
            "   - run_a1_graph_rag for athlete/representation queries\n"
            "   - run_a0_vector_rag for direct lookups\n"
            "2. Conclude with the grounded candidate answer."
        )
        return create_deep_agent(
            model=self.model,
            system_prompt=system_prompt,
            tools=self._build_harness_tools(),
            skills=[f"/{self.skills_dir.name}/"],
            backend=self.backend,
        )

    def _execute_question_analysis_skill(
        self, question: str, state: RunState
    ) -> dict[str, Any]:
        """Skill 1: Question Analysis Skill (skills/question-analysis/SKILL.md)."""
        HarnessReducer.apply_event(
            state, "SKILL_ACTIVATED", {"skill": "question-analysis"}
        )
        q_lower = question.lower()

        # 1. Normalized Entity / Venue / Date Extraction
        v_match = re.search(
            r"held at (.*?)(?: on (.*?)(?: at the|\?|$)| in \b\d{4}\b|"
            r" in the| at the|\?|$)",
            question,
            re.IGNORECASE,
        )
        venue_cue = None
        date_cue = None
        if v_match:
            venue_cue = v_match.group(1).strip(" -–\t\n?")
            if v_match.group(2):
                date_cue = v_match.group(2).strip(" -–\t\n?")

        # 2. Temporal extraction
        y_match = re.search(r"\b(19\d\d|20\d\d)\b", question)
        year_val = int(y_match.group(1)) if y_match else None

        # 3. Intent Classification
        is_nation_agg = (
            "how many nations" in q_lower
            or "which country" in q_lower
            or "which nation" in q_lower
        )
        is_relational = any(
            p in q_lower
            for p in [
                "represented",
                "competed",
                "participated",
                "won the gold medal in the event",
            ]
        )
        is_venue = bool(venue_cue or "held at" in q_lower or "venue" in q_lower)

        analysis = {
            "venue_cue": venue_cue,
            "date_cue": date_cue,
            "year": year_val,
            "is_nation_agg": is_nation_agg,
            "is_relational": is_relational,
            "is_venue_multihop": is_venue,
        }
        HarnessReducer.apply_event(
            state,
            "SKILL_EXECUTED",
            {"skill": "question-analysis", "analysis": analysis},
        )
        return analysis

    def _execute_retrieval_strategy_selection_skill(
        self, analysis: dict[str, Any], state: RunState
    ) -> AgentStrategy:
        """Skill 2: Strategy Selection (retrieval-strategy-selection/SKILL.md)."""
        HarnessReducer.apply_event(
            state, "SKILL_ACTIVATED", {"skill": "retrieval-strategy-selection"}
        )

        if analysis["is_venue_multihop"] or analysis["is_nation_agg"]:
            strat = AgentStrategy.A2_DETERMINISTIC_GRAPH
        elif analysis["is_relational"]:
            strat = AgentStrategy.A1_ADAPTIVE_GRAPHRAG
        else:
            strat = AgentStrategy.A0_VECTOR_RAG

        HarnessReducer.apply_event(
            state, "STRATEGY_CHOSEN", {"strategy": strat.value}
        )
        HarnessReducer.apply_event(
            state,
            "SKILL_EXECUTED",
            {"skill": "retrieval-strategy-selection", "selected": strat.value},
        )
        return strat

    def _execute_graph_reasoning_skill(
        self,
        question: str,
        analysis: dict[str, Any],
        state: RunState,
        a2_tools_called: list[str],
    ) -> tuple[list[str], str]:
        """Skill 3: Graph Reasoning Skill (skills/graph-reasoning/SKILL.md)."""
        HarnessReducer.apply_event(
            state, "SKILL_ACTIVATED", {"skill": "graph-reasoning"}
        )
        a2_tools_called.append("execute_composite_multihop")
        HarnessReducer.apply_event(
            state, "TOOL_CALLED", {"tool": "execute_composite_multihop"}
        )

        venue_id = None
        if analysis["venue_cue"]:
            from tgh.ingestion.graph_extractor import make_venue_id

            venue_id = make_venue_id(analysis["venue_cue"])

        comp_res = self.graph_tools.execute_composite_multihop(
            venue_id=venue_id,
            year=analysis["year"],
            date_cue=analysis.get("date_cue"),
        )

        ranked_docs: list[str] = []
        candidate_answer = ""
        if comp_res.success and comp_res.data:
            for ev in comp_res.data.get("events", []):
                ranked_docs.append(ev.event_id)
                HarnessReducer.apply_event(
                    state,
                    "EVIDENCE_ADDED",
                    {
                        "source_type": "graph_fact",
                        "source_id": f"Event:{ev.event_id}",
                        "document_id": ev.event_id,
                        "text_or_fact": (
                            f"Event {ev.name} ({ev.year}): {ev.description}"
                        ),
                        "provenance_path": ev.provenance_path,
                        "relation": "HELD_AT",
                    },
                )
            for ctx in comp_res.data.get("event_contexts", []):
                for p in ctx.participants:
                    candidate_answer = candidate_answer or p["name"]
                    HarnessReducer.apply_event(
                        state,
                        "EVIDENCE_ADDED",
                        {
                            "source_type": "graph_fact",
                            "source_id": f"{p['type']}:{p['id']}",
                            "document_id": ctx.event_id,
                            "text_or_fact": (
                                f"{p['type']} {p['name']} participated "
                                f"in event {ctx.event_id}"
                            ),
                            "relation": "PARTICIPATED_IN",
                        },
                    )

        # Fallback to vector retrieval if graph produced 0 candidate events
        if not ranked_docs:
            seeds = self.vector_retriever.retrieve_seeds(question, top_k=20)
            ranked_docs = [s.doc_id for s in seeds if s.doc_id]
            for s in seeds[:5]:
                HarnessReducer.apply_event(
                    state,
                    "EVIDENCE_ADDED",
                    {
                        "source_type": "vector_chunk",
                        "source_id": s.chunk_id,
                        "document_id": s.doc_id,
                        "text_or_fact": s.text or f"Chunk {s.chunk_id}",
                        "confidence": s.similarity,
                    },
                )

        HarnessReducer.apply_event(
            state,
            "SKILL_EXECUTED",
            {"skill": "graph-reasoning", "events_isolated": len(ranked_docs)},
        )
        return ranked_docs, candidate_answer

    def _execute_evidence_verification_skill(
        self,
        question: str,
        candidate_answer: str,
        state: RunState,
        unique_ranked: list[str],
    ) -> tuple[VerificationResult, bool, list[str]]:
        """Skill 4: Evidence Verification (skills/evidence-verification/SKILL.md)."""
        HarnessReducer.apply_event(
            state, "SKILL_ACTIVATED", {"skill": "evidence-verification"}
        )
        verification = self.verifier.verify(
            question=question,
            candidate_answer=candidate_answer,
            ledger=state.ledger,
        )
        repair_used = False

        if (
            verification.status == VerificationStatus.REPAIR_REQUIRED
            and state.repair_count < state.budget.max_repairs
        ):
            HarnessReducer.apply_event(
                state,
                "REPAIR_ATTEMPTED",
                {"reason": verification.repair_focus},
            )
            repair_ok, repair_msg = self.repairer.attempt_repair(
                question=question,
                verification_result=verification,
                ledger=state.ledger,
                repair_budget=1,
            )
            repair_used = True

            # Re-rank with newly discovered documents from repair
            new_doc_ids = state.ledger.get_document_ids()
            combined_ranked: list[str] = []
            for did in new_doc_ids + unique_ranked:
                if did not in combined_ranked:
                    combined_ranked.append(did)
            unique_ranked = combined_ranked
            HarnessReducer.apply_event(
                state, "DOCS_RANKED", {"ranked_doc_ids": unique_ranked}
            )

            # Re-verify post-repair
            verification = self.verifier.verify(
                question=question,
                candidate_answer=candidate_answer,
                ledger=state.ledger,
            )

        HarnessReducer.apply_event(
            state,
            "SKILL_EXECUTED",
            {
                "skill": "evidence-verification",
                "status": verification.status.value,
                "repair_used": repair_used,
            },
        )
        return verification, repair_used, unique_ranked

    def _execute_answer_synthesis_skill(
        self,
        candidate_answer: str,
        verification: VerificationResult,
        state: RunState,
    ) -> str:
        """Skill 5: Answer Synthesis Skill (skills/answer-synthesis/SKILL.md)."""
        HarnessReducer.apply_event(
            state, "SKILL_ACTIVATED", {"skill": "answer-synthesis"}
        )
        grounded_answer = candidate_answer
        if (
            verification.status == VerificationStatus.INSUFFICIENT
            and not state.ledger.list_items()
        ):
            grounded_answer = "INSUFFICIENT_EVIDENCE"

        HarnessReducer.apply_event(
            state,
            "SKILL_EXECUTED",
            {"skill": "answer-synthesis", "final_answer": grounded_answer},
        )
        return grounded_answer

    def route_question(self, question: str) -> AgentStrategy:
        """Select reasoning strategy via Question Analysis and Strategy Selection."""
        temp_state = RunState(question=question)
        analysis = self._execute_question_analysis_skill(question, temp_state)
        return self._execute_retrieval_strategy_selection_skill(analysis, temp_state)

    def run(
        self,
        question: str,
        qtype: str | None = None,
        budget: ExecutionBudget | None = None,
    ) -> AgenticOrchestrationResult:
        """Execute full Agentic GraphRAG reasoning loop under harness oversight."""
        t0 = time.perf_counter()
        state = RunState(question=question, budget=budget)
        mcp_tools_called: list[str] = []
        a2_tools_called: list[str] = []

        model_invoked = False
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        model_latency_ms = 0.0
        model_answer: str | None = None

        if self.deep_agent is not None:
            self._active_run_state = state
            self._active_mcp_tools = mcp_tools_called
            self._active_a2_tools = a2_tools_called
            t_model_start = time.perf_counter()
            try:
                res = self.deep_agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Please investigate and answer this "
                                    f"Olympic query: {question}\n"
                                    "1. Use investigate_and_retrieve with the "
                                    "appropriate strategy.\n"
                                    "2. Use verify_and_repair to verify "
                                    "the evidence.\n"
                                    "3. State the final grounded answer."
                                ),
                            }
                        ]
                    }
                )
                model_latency_ms = (time.perf_counter() - t_model_start) * 1000.0
                model_invoked = True

                messages = res.get("messages", []) if isinstance(res, dict) else []
                for m in messages:
                    um = getattr(m, "usage_metadata", None)
                    if um:
                        input_tokens += um.get("input_tokens", 0)
                        output_tokens += um.get("output_tokens", 0)
                        total_tokens += um.get("total_tokens", 0)
                    if getattr(m, "type", None) == "ai" and getattr(
                        m, "content", None
                    ):
                        if isinstance(m.content, str):
                            model_answer = m.content
                        elif isinstance(m.content, list):
                            text_parts = [
                                p.get("text", "")
                                for p in m.content
                                if isinstance(p, dict) and "text" in p
                            ]
                            if text_parts:
                                model_answer = " ".join(text_parts)
            except Exception as e:
                logger.warning(
                    f"DeepAgents model invocation encountered an error: {e}. "
                    "Executing deterministic harness fallback."
                )
                HarnessReducer.apply_event(
                    state, "MODEL_FALLBACK", {"error": str(e)}
                )
            finally:
                self._active_run_state = None
                self._active_mcp_tools = None
                self._active_a2_tools = None

        # Fallback / deterministic guarantee if model didn't execute retrieval
        if state.phase == ExecutionPhase.INITIALIZED:
            # 1. Skill 1: Question Analysis
            analysis = self._execute_question_analysis_skill(question, state)

            # 2. Skill 2: Retrieval Strategy Selection
            strategy = self._execute_retrieval_strategy_selection_skill(
                analysis, state
            )

            ranked_docs: list[str] = []
            candidate_answer = ""

            # 3. Strategy Execution (with Skill 3: Graph Reasoning where appropriate)
            if strategy == AgentStrategy.A2_DETERMINISTIC_GRAPH:
                ranked_docs, candidate_answer = (
                    self._execute_graph_reasoning_skill(
                        question, analysis, state, a2_tools_called
                    )
                )

            elif strategy == AgentStrategy.A1_ADAPTIVE_GRAPHRAG:
                rag_res = self.graphrag_retriever.retrieve(
                    question, top_k_seeds=20
                )
                ranked_docs = rag_res.ranked_doc_ids
                for ec in rag_res.evidence_chunks[:10]:
                    HarnessReducer.apply_event(
                        state,
                        "EVIDENCE_ADDED",
                        {
                            "source_type": (
                                "vector_chunk"
                                if ec.source == "seed"
                                else "graph_fact"
                            ),
                            "source_id": ec.chunk_id,
                            "document_id": ec.doc_id,
                            "text_or_fact": (
                                f"Evidence for {ec.doc_id} score={ec.score:.4f}"
                            ),
                            "provenance_path": (
                                ec.provenance[0] if ec.provenance else None
                            ),
                            "confidence": ec.score,
                        },
                    )
                if rag_res.evidence_chunks:
                    candidate_answer = rag_res.evidence_chunks[0].doc_id

            else:
                # A0 Vector RAG
                seeds = self.vector_retriever.retrieve_seeds(question, top_k=20)
                ranked_docs = [s.doc_id for s in seeds if s.doc_id]
                for s in seeds[:10]:
                    HarnessReducer.apply_event(
                        state,
                        "EVIDENCE_ADDED",
                        {
                            "source_type": "vector_chunk",
                            "source_id": s.chunk_id,
                            "document_id": s.doc_id,
                            "text_or_fact": s.text or f"Chunk {s.chunk_id}",
                            "confidence": s.similarity,
                        },
                    )
                if ranked_docs:
                    candidate_answer = ranked_docs[0]

            # Deduplicate and register ranked documents
            seen: set[str] = set()
            unique_ranked: list[str] = []
            for d in ranked_docs:
                if d and d not in seen:
                    seen.add(d)
                    unique_ranked.append(d)

            HarnessReducer.apply_event(
                state, "DOCS_RANKED", {"ranked_doc_ids": unique_ranked}
            )
            HarnessReducer.apply_event(
                state, "ANSWER_PROPOSED", {"answer": candidate_answer}
            )

        # 4. Skill 4: Evidence Verification & Bounded Repair (ensure it runs)
        repair_used = state.repair_count > 0
        unique_ranked = state.ranked_doc_ids
        cand_ans = state.candidate_answer or ""
        if "evidence-verification" not in state.skills_invoked:
            verification, repair_used, unique_ranked = (
                self._execute_evidence_verification_skill(
                    question, cand_ans, state, unique_ranked
                )
            )
        else:
            verification = self.verifier.verify(
                question=question,
                candidate_answer=cand_ans,
                ledger=state.ledger,
            )

        # 5. Skill 5: Answer Synthesis
        final_answer = self._execute_answer_synthesis_skill(
            cand_ans, verification, state
        )

        HarnessReducer.apply_event(
            state, "EXECUTION_COMPLETED", {"ranked_count": len(unique_ranked)}
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return AgenticOrchestrationResult(
            run_id=state.run_id,
            question=question,
            strategy=state.strategy or AgentStrategy.A0_VECTOR_RAG,
            state_view=state.get_view(),
            ranked_doc_ids=unique_ranked,
            candidate_answer=final_answer,
            verification=verification,
            repair_used=repair_used,
            latency_ms=elapsed_ms,
            mcp_tools_invoked=mcp_tools_called,
            a2_tools_invoked=a2_tools_called,
            evidence_items=state.ledger.get_all(),
            skills_invoked=list(state.skills_invoked),
            model_name=self.model_name if model_invoked else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_latency_ms=model_latency_ms,
            model_answer=model_answer,
        )
