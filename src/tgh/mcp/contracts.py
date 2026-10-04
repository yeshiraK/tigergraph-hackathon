"""Typed data models and contracts for Layer 3 deterministic graph tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tgh.telemetry.trace import OperationTrace


@dataclass(frozen=True)
class EventCandidate:
    """A candidate Event vertex with attributes and provenance."""

    event_id: str
    name: str
    year: int
    description: str
    provenance_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EventContext:
    """Detailed domain context connected to an Event."""

    event_id: str
    name: str
    year: int
    description: str
    sport: dict[str, str] | None = None  # {"sport_id": ..., "name": ...}
    venue: dict[str, str] | None = None  # {"venue_id": ..., "name": ...}
    participants: list[dict[str, str]] = field(
        default_factory=list
    )  # [{"id": ..., "name": ..., "type": ...}]
    provenance_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CountryRepresentation:
    """Representation link between a participant and Country."""

    participant_type: str  # "Person" or "Team"
    participant_id: str
    participant_name: str
    country_id: str
    country_name: str
    country_code: str
    provenance_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CountryAggregation:
    """Deduplicated country aggregation result."""

    distinct_count: int
    country_ids: list[str]
    country_names: list[str]
    provenance: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolExecutionResult:
    """Standardized deterministic execution envelope for all A2 tools."""

    success: bool
    data: Any
    provenance: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    trace: OperationTrace | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": (
                [item.to_dict() for item in self.data]
                if isinstance(self.data, list) and hasattr(self.data[0], "to_dict")
                else (
                    self.data.to_dict() if hasattr(self.data, "to_dict") else self.data
                )
            ),
            "provenance": self.provenance,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "trace": self.trace.to_dict() if self.trace else None,
        }
