from src.observability.tracer import InvestigationTracer


def test_tracer_starts_run():

    tracer = InvestigationTracer()

    run_id = tracer.start_run(
        question="Why is instance unhealthy?",
        capability="instance_health",
    )

    assert run_id
    assert tracer.run_id == run_id

    events = tracer.get_events()

    assert len(events) == 1
    assert events[0]["event_type"] == "run_started"


def test_node_execution_is_recorded():

    tracer = InvestigationTracer()

    tracer.start_run(
        question="Why is instance unhealthy?"
    )

    start = tracer.node_started(
        "investigate",
        iteration=1,
    )

    tracer.node_completed(
        "investigate",
        start,
    )

    events = tracer.get_events()

    event_types = [
        event["event_type"]
        for event in events
    ]

    assert "node_started" in event_types
    assert "node_completed" in event_types


def test_tool_execution_is_recorded():

    tracer = InvestigationTracer()

    tracer.start_run(
        question="Why is instance unhealthy?"
    )

    start = tracer.tool_started(
        "get_instance_health",
        {
            "instance_id": "i-demo-001"
        },
    )

    tracer.tool_completed(
        "get_instance_health",
        start,
    )

    events = tracer.get_events()

    assert any(
        event["event_type"]
        == "tool_started"
        for event in events
    )

    assert any(
        event["event_type"]
        == "tool_completed"
        for event in events
    )


def test_decision_is_recorded():

    tracer = InvestigationTracer()

    tracer.start_run(
        question="Why is instance unhealthy?"
    )

    tracer.decision(
        decision="investigate",
        reason="Additional evidence required",
        iteration=1,
    )

    events = tracer.get_events()

    decision_events = [
        event
        for event in events
        if event["event_type"]
        == "decision"
    ]

    assert len(decision_events) == 1
    assert (
        decision_events[0]["data"]["decision"]
        == "investigate"
    )


def test_error_is_recorded():

    tracer = InvestigationTracer()

    tracer.start_run(
        question="Why is instance unhealthy?"
    )

    tracer.error(
        "tool_execution",
        "Connection failed",
    )

    events = tracer.get_events()

    errors = [
        event
        for event in events
        if event["event_type"]
        == "error"
    ]

    assert len(errors) == 1
    assert (
        errors[0]["data"]["error"]
        == "Connection failed"
    )


def test_summary():

    tracer = InvestigationTracer()

    tracer.start_run(
        question="Why is instance unhealthy?"
    )

    start = tracer.node_started(
        "investigate",
        iteration=1,
    )

    tracer.node_completed(
        "investigate",
        start,
    )

    tracer.end_run(
        "completed"
    )

    summary = tracer.summary()

    assert summary["run_id"]
    assert summary["event_count"] >= 3
    assert summary["node_count"] == 1
    assert summary["error_count"] == 0