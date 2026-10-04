"""Lightweight execution tracing for A2/A3/A4 agent operations."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OperationTrace:
    """Structured record of an individual operation or tool execution."""

    run_id: str
    operation_name: str
    input_summary: dict[str, Any]
    start_time: float
    end_time: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    result_summary: dict[str, Any] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    error: str | None = None

    def finish(
        self,
        success: bool = True,
        result_summary: dict[str, Any] | None = None,
        provenance: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        """Mark the trace as finished and compute latency."""
        self.end_time = time.perf_counter()
        self.latency_ms = (self.end_time - self.start_time) * 1000.0
        self.success = success
        if result_summary is not None:
            self.result_summary = result_summary
        if provenance is not None:
            self.provenance = provenance
        if error is not None:
            self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to a JSON-serializable dictionary."""
        return asdict(self)


class TraceRecorder:
    """Thread-safe collector for operation traces."""

    def __init__(self, run_id: str | None = None) -> None:
        """Initialize recorder with a unique run_id."""
        self.run_id = run_id or str(uuid.uuid4())
        self._traces: list[OperationTrace] = []

    def start_trace(
        self,
        operation_name: str,
        input_summary: dict[str, Any] | None = None,
    ) -> OperationTrace:
        """Create and start a new operation trace."""
        trace = OperationTrace(
            run_id=self.run_id,
            operation_name=operation_name,
            input_summary=input_summary or {},
            start_time=time.perf_counter(),
        )
        self._traces.append(trace)
        return trace

    def get_traces(self) -> list[OperationTrace]:
        """Return all recorded traces."""
        return list(self._traces)

    def to_list(self) -> list[dict[str, Any]]:
        """Return all traces as serializable dictionaries."""
        return [t.to_dict() for t in self._traces]
