"""Evidence Ledger and Grounding models for Layer 4 & Layer 5."""

import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceSourceType(StrEnum):
    """Categorical source type for accumulated evidence."""

    VECTOR_CHUNK = "vector_chunk"
    GRAPH_FACT = "graph_fact"
    GRAPH_PATH = "graph_path"


@dataclass(frozen=True)
class EvidenceItem:
    """An individual discrete evidence unit supporting an answer candidate."""

    evidence_id: str
    source_type: EvidenceSourceType
    source_id: str  # chunk_id, vertex_id, or path signature
    document_id: str | None = None
    text_or_fact: str = ""
    provenance_path: str | None = None
    relation: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["source_type"] = self.source_type.value
        return res


class EvidenceLedger:
    """Authoritative ledger controlling what facts and documents support an answer."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or str(uuid.uuid4())
        self._items: dict[str, EvidenceItem] = {}
        self._by_source_id: dict[str, str] = {}  # source_id -> evidence_id

    def add_item(
        self,
        source_type: EvidenceSourceType | str,
        source_id: str,
        document_id: str | None = None,
        text_or_fact: str = "",
        provenance_path: str | None = None,
        relation: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceItem:
        """Add or update an evidence item in the ledger."""
        st = (
            source_type
            if isinstance(source_type, EvidenceSourceType)
            else EvidenceSourceType(source_type)
        )
        if source_id in self._by_source_id:
            # Existing item: return existing
            eid = self._by_source_id[source_id]
            return self._items[eid]

        evidence_id = f"ev-{len(self._items) + 1:04d}-{str(uuid.uuid4())[:6]}"
        item = EvidenceItem(
            evidence_id=evidence_id,
            source_type=st,
            source_id=source_id,
            document_id=document_id,
            text_or_fact=text_or_fact,
            provenance_path=provenance_path,
            relation=relation,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._items[evidence_id] = item
        self._by_source_id[source_id] = evidence_id
        return item

    def get_item(self, evidence_id: str) -> EvidenceItem | None:
        return self._items.get(evidence_id)

    def list_items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def get_all(self) -> list[EvidenceItem]:
        return self.list_items()

    def get_document_ids(self) -> list[str]:
        """Return unique document IDs present in the evidence ledger."""
        doc_ids: list[str] = []
        for item in self._items.values():
            if item.document_id and item.document_id not in doc_ids:
                doc_ids.append(item.document_id)
        return doc_ids

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items.values()]
