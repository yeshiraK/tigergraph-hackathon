"""Layer 5: A3 Evidence-Grounded Verification and Bounded One-Repair Layer."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from tgh.evidence.ledger import EvidenceItem, EvidenceLedger
from tgh.mcp.graph_tools import TigerGraphTools
from tgh.retrieval.graphrag import GraphRAGRetriever
from tgh.telemetry.trace import TraceRecorder


class VerificationStatus(StrEnum):
    """Categorical verification state."""

    SUPPORTED = "SUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTED = "CONTRADICTED"
    REPAIR_REQUIRED = "REPAIR_REQUIRED"


@dataclass(frozen=True)
class SufficiencyCheckResult:
    """Breakdown of entity/constraint sufficiency checks."""

    is_sufficient: bool
    missing_aspects: list[str]
    detected_entities: list[str]
    detected_years: list[int]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    """Structured evaluation output of candidate answer verification."""

    status: VerificationStatus
    candidate_answer: str
    sufficiency: SufficiencyCheckResult
    supporting_evidence_ids: list[str]
    repair_needed: bool
    repair_focus: str | None = None
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["status"] = self.status.value
        return res


def _normalize(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    cleaned = re.sub(r"[^\w\s-]", " ", norm, flags=re.UNICODE).lower()
    return re.sub(r"\s+", " ", cleaned).strip()


class EvidenceVerifier:
    """Deterministic, grounded sufficiency and contradiction verifier for A3."""

    def __init__(self, recorder: TraceRecorder | None = None) -> None:
        self.recorder = recorder or TraceRecorder()

    def check_sufficiency(
        self,
        question: str,
        candidate_answer: str,
        evidence_items: list[EvidenceItem],
    ) -> SufficiencyCheckResult:
        """Inspect whether evidence items collectively cover question constraints."""
        reasons: list[str] = []
        missing_aspects: list[str] = []

        norm_ans = _normalize(candidate_answer)

        # 1. Year constraint check
        years_in_q = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", question)]
        years_in_ev: list[int] = []
        for ev in evidence_items:
            for y_str in re.findall(r"\b(19\d\d|20\d\d)\b", ev.text_or_fact):
                years_in_ev.append(int(y_str))

        for req_year in years_in_q:
            if req_year not in years_in_ev:
                missing_aspects.append(f"year_{req_year}")
                reasons.append(f"Required year {req_year} not evidenced in facts.")

        # 2. Candidate Answer grounding check
        ans_grounded = False
        norm_tokens = [t for t in norm_ans.split() if len(t) > 2]
        for ev in evidence_items:
            norm_fact = _normalize(ev.text_or_fact)
            if norm_ans in norm_fact:
                ans_grounded = True
                break
            if norm_tokens:
                token_hits = sum(1 for tok in norm_tokens if tok in norm_fact)
                if token_hits == len(norm_tokens):
                    ans_grounded = True
                    break

        if not ans_grounded and candidate_answer:
            missing_aspects.append("answer_unsupported")
            reasons.append(
                f"Candidate answer '{candidate_answer}' is not supported by "
                "any accumulated evidence item."
            )

        # 3. Minimum evidence volume
        if not evidence_items:
            missing_aspects.append("empty_evidence")
            reasons.append("Evidence ledger is completely empty.")

        is_sufficient = len(missing_aspects) == 0
        return SufficiencyCheckResult(
            is_sufficient=is_sufficient,
            missing_aspects=missing_aspects,
            detected_entities=norm_tokens,
            detected_years=years_in_q,
            reasons=reasons,
        )

    def verify(
        self,
        question: str,
        candidate_answer: str,
        ledger: EvidenceLedger,
    ) -> VerificationResult:
        """Run grounded verification over ledger items and return structured verdict."""
        trace = self.recorder.start_trace(
            "verify_answer",
            {"question": question, "candidate_answer": candidate_answer},
        )
        items = ledger.list_items()
        sufficiency = self.check_sufficiency(question, candidate_answer, items)

        supporting_ids = [
            item.evidence_id
            for item in items
            if _normalize(candidate_answer) in _normalize(item.text_or_fact)
        ]

        if sufficiency.is_sufficient:
            status = VerificationStatus.SUPPORTED
            repair_needed = False
            repair_focus = None
            explanation = "Candidate answer is fully supported by evidence."
        elif "answer_unsupported" in sufficiency.missing_aspects:
            status = VerificationStatus.REPAIR_REQUIRED
            repair_needed = True
            repair_focus = "retrieve_missing_gold_document"
            explanation = "Answer lacks grounded evidence; repair required."
        elif any(m.startswith("year_") for m in sufficiency.missing_aspects):
            status = VerificationStatus.REPAIR_REQUIRED
            repair_needed = True
            repair_focus = "filter_by_year_constraint"
            explanation = "Year constraint not satisfied in evidence; repair required."
        else:
            status = VerificationStatus.INSUFFICIENT
            repair_needed = False
            repair_focus = None
            explanation = f"Insufficient evidence: {'; '.join(sufficiency.reasons)}"

        res = VerificationResult(
            status=status,
            candidate_answer=candidate_answer,
            sufficiency=sufficiency,
            supporting_evidence_ids=supporting_ids,
            repair_needed=repair_needed,
            repair_focus=repair_focus,
            explanation=explanation,
        )
        trace.finish(
            success=True,
            result_summary={"status": status.value, "repair_needed": repair_needed},
        )
        return res


class BoundedRepairExecutor:
    """Executes exactly ONE repair attempt when verification flags REPAIR_REQUIRED."""

    def __init__(
        self,
        tools: TigerGraphTools | None = None,
        graphrag_retriever: GraphRAGRetriever | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self.tools = tools
        self.graphrag = graphrag_retriever
        self.recorder = recorder or TraceRecorder()

    def attempt_repair(
        self,
        question: str,
        verification_result: VerificationResult,
        ledger: EvidenceLedger,
        repair_budget: int = 1,
    ) -> tuple[bool, str]:
        """Perform a single targeted repair and record newly recovered evidence."""
        trace = self.recorder.start_trace(
            "attempt_repair",
            {
                "focus": verification_result.repair_focus,
                "missing": verification_result.sufficiency.missing_aspects,
            },
        )
        if repair_budget <= 0:
            err = "Repair budget exhausted (max 1 repair attempt permitted)."
            trace.finish(success=False, error=err)
            return False, err

        focus = verification_result.repair_focus or "retrieve_missing_gold_document"
        recovered_count = 0

        # Targeted repair strategy 1: Venue / Participant composite traversal via A2
        if (
            self.tools
            and ("venue" in question.lower() or "held at" in question.lower())
        ):
            # Extract venue name hint
            v_match = re.search(
                r"held at ([^,\?]+?)(?: on| in \b\d{4}\b| in the| at the|\?|$)",
                question,
                re.IGNORECASE,
            )
            y_match = re.search(r"\b(19\d\d|20\d\d)\b", question)
            year_val = int(y_match.group(1)) if y_match else None

            if v_match:
                from tgh.ingestion.graph_extractor import make_venue_id

                raw_vname = v_match.group(1).strip(" -–\t\n")
                venue_id = make_venue_id(raw_vname)
                comp_res = self.tools.execute_composite_multihop(
                    venue_id=venue_id,
                    year=year_val,
                )
                if comp_res.success and comp_res.data:
                    for ev in comp_res.data.get("events", []):
                        ledger.add_item(
                            source_type="graph_fact",
                            source_id=f"Event:{ev.event_id}",
                            document_id=ev.event_id,
                            text_or_fact=(
                                f"Event {ev.name} ({ev.year}): {ev.description}"
                            ),
                            provenance_path=ev.provenance_path,
                            relation="HELD_AT",
                        )
                        recovered_count += 1
                    for ctx in comp_res.data.get("event_contexts", []):
                        for p in ctx.participants:
                            ledger.add_item(
                                source_type="graph_fact",
                                source_id=f"{p['type']}:{p['id']}",
                                document_id=ctx.event_id,
                                text_or_fact=(
                                    f"{p['type']} {p['name']} participated "
                                    f"in event {ctx.event_id}"
                                ),
                                relation="PARTICIPATED_IN",
                            )
                            recovered_count += 1

        # Targeted repair strategy 2: GraphRAG fallback if A2 didn't fire or returned 0
        if recovered_count == 0 and self.graphrag:
            rag_res = self.graphrag.retrieve(question, top_k_seeds=10)
            for ec in rag_res.evidence_chunks:
                ledger.add_item(
                    source_type="vector_chunk" if ec.source == "seed" else "graph_fact",
                    source_id=ec.chunk_id,
                    document_id=ec.doc_id,
                    text_or_fact=f"Evidence for {ec.doc_id} score={ec.score:.4f}",
                    provenance_path=ec.provenance[0] if ec.provenance else None,
                    confidence=ec.score,
                )
                recovered_count += 1

        success = recovered_count > 0
        summary_msg = (
            f"Repair completed: recovered {recovered_count} additional evidence items."
        )
        trace.finish(
            success=success,
            result_summary={"recovered_count": recovered_count, "focus": focus},
        )
        return success, summary_msg
