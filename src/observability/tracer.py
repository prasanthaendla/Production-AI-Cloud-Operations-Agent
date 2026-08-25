"""
Lightweight observability and tracing for the AI Cloud Operations Agent.

Stage 17:
- Tracks investigation runs.
- Tracks node execution.
- Tracks tool execution.
- Tracks duration.
- Tracks errors.
- Keeps observability independent from investigation intelligence.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class InvestigationTracer:
    """
    Lightweight in-process tracer.

    This is intentionally simple for Stage 17.
    Later it can be connected to CloudWatch, OpenTelemetry,
    LangSmith, or another production observability backend
    without changing the investigation architecture.
    """

    def __init__(self) -> None:
        self.run_id: str = ""
        self.started_at: str = ""
        self.events: List[Dict[str, Any]] = []

    # --------------------------------------------------------------
    # Run
    # --------------------------------------------------------------

    def start_run(
        self,
        question: str,
        capability: str = "",
    ) -> str:

        self.run_id = str(uuid.uuid4())

        self.started_at = self._timestamp()

        self.events = []

        self._record(
            event_type="run_started",
            data={
                "question": question,
                "capability": capability,
            },
        )

        return self.run_id

    def end_run(
        self,
        status: str,
    ) -> None:

        self._record(
            event_type="run_completed",
            data={
                "status": status,
            },
        )

    # --------------------------------------------------------------
    # Nodes
    # --------------------------------------------------------------

    def node_started(
        self,
        node_name: str,
        iteration: Optional[int] = None,
    ) -> float:

        start_time = time.perf_counter()

        self._record(
            event_type="node_started",
            data={
                "node": node_name,
                "iteration": iteration,
            },
        )

        return start_time

    def node_completed(
        self,
        node_name: str,
        start_time: float,
        status: str = "completed",
    ) -> None:

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        self._record(
            event_type="node_completed",
            data={
                "node": node_name,
                "status": status,
                "duration_ms": round(
                    duration_ms,
                    2,
                ),
            },
        )

    # --------------------------------------------------------------
    # Tools
    # --------------------------------------------------------------

    def tool_started(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> float:

        start_time = time.perf_counter()

        self._record(
            event_type="tool_started",
            data={
                "tool": tool_name,
                "arguments": arguments,
            },
        )

        return start_time

    def tool_completed(
        self,
        tool_name: str,
        start_time: float,
        status: str = "completed",
    ) -> None:

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        self._record(
            event_type="tool_completed",
            data={
                "tool": tool_name,
                "status": status,
                "duration_ms": round(
                    duration_ms,
                    2,
                ),
            },
        )

    # --------------------------------------------------------------
    # Decisions
    # --------------------------------------------------------------

    def decision(
        self,
        decision: str,
        reason: str = "",
        iteration: Optional[int] = None,
    ) -> None:

        self._record(
            event_type="decision",
            data={
                "decision": decision,
                "reason": reason,
                "iteration": iteration,
            },
        )

    # --------------------------------------------------------------
    # Errors
    # --------------------------------------------------------------

    def error(
        self,
        component: str,
        error: Exception | str,
    ) -> None:

        self._record(
            event_type="error",
            data={
                "component": component,
                "error": str(error),
            },
        )

    # --------------------------------------------------------------
    # Evidence
    # --------------------------------------------------------------

    def evidence_recorded(
        self,
        tool_name: str,
        evidence_count: int,
    ) -> None:

        self._record(
            event_type="evidence_recorded",
            data={
                "tool": tool_name,
                "evidence_count": evidence_count,
            },
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:

        node_events = [
            event
            for event in self.events
            if event["event_type"]
            == "node_completed"
        ]

        tool_events = [
            event
            for event in self.events
            if event["event_type"]
            == "tool_completed"
        ]

        errors = [
            event
            for event in self.events
            if event["event_type"]
            == "error"
        ]

        total_duration_ms = 0.0

        for event in node_events:
            total_duration_ms += float(
                event["data"].get(
                    "duration_ms",
                    0,
                )
            )

        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "event_count": len(self.events),
            "node_count": len(node_events),
            "tool_count": len(tool_events),
            "error_count": len(errors),
            "total_node_duration_ms": round(
                total_duration_ms,
                2,
            ),
        }

    # --------------------------------------------------------------
    # Export
    # --------------------------------------------------------------

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def to_json(self) -> str:

        return json.dumps(
            {
                "run_id": self.run_id,
                "events": self.events,
                "summary": self.summary(),
            },
            indent=2,
            default=str,
        )

    # --------------------------------------------------------------
    # Internal
    # --------------------------------------------------------------

    def _record(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:

        self.events.append(
            {
                "timestamp": self._timestamp(),
                "run_id": self.run_id,
                "event_type": event_type,
                "data": data,
            }
        )

    @staticmethod
    def _timestamp() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()