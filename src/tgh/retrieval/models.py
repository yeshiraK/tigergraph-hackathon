"""Data models for deterministic GraphRAG retrieval."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SeedChunk:
    """A seed chunk retrieved directly via vector search."""

    chunk_id: str
    doc_id: str
    distance: float
    similarity: float
    rank: int
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveredEntity:
    """An entity vertex reached via MENTIONS edge from a chunk."""

    entity_id: str
    seed_chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DomainVertex:
    """A domain-specific vertex reached via RESOLVES_TO (Person, Venue, etc.)."""

    vertex_type: str
    vertex_id: str
    resolved_from_entity_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphPath:
    """A single edge traversal step in graph exploration."""

    source_type: str
    source_id: str
    edge_type: str
    target_type: str
    target_id: str
    hop: int

    def to_str(self) -> str:
        return (
            f"({self.source_type}:{self.source_id}) -[{self.edge_type}]-> "
            f"({self.target_type}:{self.target_id})"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceChunk:
    """A chunk included as retrieval evidence with full provenance."""

    chunk_id: str
    doc_id: str
    score: float
    source: str  # "seed" or "graph"
    hop_distance: int
    provenance: list[str] = field(default_factory=list)
    seed_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphRAGRetrievalResult:
    """Full structured result of a GraphRAG retrieval request."""

    question: str
    seed_chunks: list[SeedChunk]
    discovered_entities: list[DiscoveredEntity]
    discovered_domain_vertices: list[DomainVertex]
    graph_paths: list[GraphPath]
    evidence_chunks: list[EvidenceChunk]
    ranked_doc_ids: list[str]
    retrieval_strategy: str = "GraphRAG_Adaptive_v1"
    latency_ms: float = 0.0
    expansion_stats: dict[str, int] = field(default_factory=dict)
    decision_reason: str = ""
    is_graph_expanded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "retrieval_strategy": self.retrieval_strategy,
            "decision_reason": self.decision_reason,
            "is_graph_expanded": self.is_graph_expanded,
            "latency_ms": self.latency_ms,
            "expansion_stats": self.expansion_stats,
            "ranked_doc_ids": self.ranked_doc_ids,
            "seed_chunks": [s.to_dict() for s in self.seed_chunks],
            "discovered_entities": [e.to_dict() for e in self.discovered_entities],
            "discovered_domain_vertices": [
                v.to_dict() for v in self.discovered_domain_vertices
            ],
            "graph_paths": [p.to_dict() for p in self.graph_paths],
            "evidence_chunks": [ec.to_dict() for ec in self.evidence_chunks],
        }
