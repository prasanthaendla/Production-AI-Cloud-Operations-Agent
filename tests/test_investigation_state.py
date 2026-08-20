"""
Tests for InvestigationState.
"""

from src.agent.investigation_state import (
    InvestigationState,
)


def test_investigation_state_initialization():

    state = InvestigationState(
        question=(
            "Why is instance i-demo-001 unhealthy?"
        ),
        capability="instance_health",
    )

    assert (
        state.question
        == "Why is instance i-demo-001 unhealthy?"
    )

    assert (
        state.capability
        == "instance_health"
    )

    assert state.iterations == 0

    assert state.tool_calls == []

    assert state.evidence == []

    assert state.findings == []


def test_iteration_tracking():

    state = InvestigationState(
        question="Why is my instance unhealthy?",
        capability="instance_health",
    )

    state.record_iteration(2)

    assert state.iterations == 2


def test_tool_call_tracking():

    state = InvestigationState(
        question="Why is my instance unhealthy?",
        capability="instance_health",
    )

    state.record_tool_call(
        "get_instance_health",
        {
            "instance_id": "i-demo-001"
        },
    )

    assert len(state.tool_calls) == 1

    assert (
        state.tool_calls[0]["tool"]
        == "get_instance_health"
    )


def test_evidence_tracking():

    state = InvestigationState(
        question="Why is my instance unhealthy?",
        capability="instance_health",
    )

    state.record_evidence(
        "get_instance_health",
        {
            "instance_id": "i-demo-001"
        },
        {
            "health": "degraded",
            "cpu_utilization": 92.4,
        },
    )

    assert len(state.evidence) == 1

    assert (
        state.evidence[0]["result"]["health"]
        == "degraded"
    )


def test_findings_tracking():

    state = InvestigationState(
        question="Why is my instance unhealthy?",
        capability="instance_health",
    )

    state.add_finding(
        "CPU utilization is high."
    )

    assert (
        state.findings
        == ["CPU utilization is high."]
    )


def test_state_summary():

    state = InvestigationState(
        question="Why is my instance unhealthy?",
        capability="instance_health",
    )

    state.record_iteration(2)

    state.record_tool_call(
        "get_instance_health",
        {
            "instance_id": "i-demo-001"
        },
    )

    state.record_evidence(
        "get_instance_health",
        {
            "instance_id": "i-demo-001"
        },
        {
            "health": "degraded"
        },
    )

    summary = state.summary()

    assert (
        summary["capability"]
        == "instance_health"
    )

    assert summary["iterations"] == 2

    assert len(summary["tool_calls"]) == 1

    assert len(summary["evidence"]) == 1