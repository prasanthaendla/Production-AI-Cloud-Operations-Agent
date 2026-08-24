"""
Tests for RootCauseAssessor.
"""

from src.agent.root_cause_assessor import (
    RootCauseAssessor,
)


def test_resource_saturation_is_supported():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "Resource saturation may be contributing to application degradation."
        }
    ]

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {
                "instance_id": "i-demo-001"
            },
            "result": {
                "cpu_utilization": 92.4,
                "memory_utilization": 81.7,
            },
        }
    ]

    result = assessor.assess(
        hypotheses,
        evidence,
    )

    assert (
        result["root_cause"]
        == hypotheses[0]["hypothesis"]
    )

    assert (
        result["score"] > 0
    )

    assert len(
        result["supporting_evidence"]
    ) == 1


def test_degraded_instance_is_supported():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "The degraded instance health may be contributing to the observed incident."
        }
    ]

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {},
            "result": {
                "health": "degraded"
            },
        }
    ]

    result = assessor.assess(
        hypotheses,
        evidence,
    )

    assert (
        result["score"] > 0
    )

    assert len(
        result["supporting_evidence"]
    ) == 1


def test_network_failure_is_supported():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "Network problems may be contributing to the incident."
        }
    ]

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {},
            "result": {
                "network_status": "unhealthy"
            },
        }
    ]

    result = assessor.assess(
        hypotheses,
        evidence,
    )

    assert (
        result["score"] > 0
    )


def test_application_errors_are_supported():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "Application errors may be contributing to the incident."
        }
    ]

    evidence = [
        {
            "tool": "get_application_logs",
            "arguments": {},
            "result": {
                "errors": [
                    "Database connection timeout",
                    "HTTP 500 error",
                ]
            },
        }
    ]

    result = assessor.assess(
        hypotheses,
        evidence,
    )

    assert (
        result["score"] > 0
    )


def test_successful_deployment_contradicts_deployment_hypothesis():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "A recent deployment may be responsible for the incident."
        }
    ]

    evidence = [
        {
            "tool": "get_recent_deployments",
            "arguments": {},
            "result": {
                "status": "SUCCESS"
            },
        }
    ]

    result = assessor.assess(
        hypotheses,
        evidence,
    )

    assert (
        len(
            result["contradicting_evidence"]
        ) == 1
    )


def test_failed_deployment_supports_deployment_hypothesis():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "A recent deployment may be responsible for the incident."
        }
    ]

    evidence = [
        {
            "tool": "get_recent_deployments",
            "arguments": {},
            "result": {
                "status": "FAILED"
            },
        }
    ]

    result = assessor.assess(
        hypotheses,
        evidence,
    )

    assert (
        result["score"] > 0
    )


def test_multiple_hypotheses_are_ranked():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "Resource saturation may be contributing to application degradation."
        },
        {
            "hypothesis":
                "Network problems may be contributing to the incident."
        },
    ]

    evidence = [
        {
            "tool": "get_instance_health",
            "arguments": {},
            "result": {
                "cpu_utilization": 92.4,
                "memory_utilization": 81.7,
                "network_status": "normal",
            },
        }
    ]

    result = assessor.assess(
        hypotheses,
        evidence,
    )

    assert (
        result["root_cause"]
        == hypotheses[0]["hypothesis"]
    )

    assert len(
        result["all_assessments"]
    ) == 2


def test_no_hypotheses_returns_uncertain_result():

    assessor = RootCauseAssessor()

    result = assessor.assess(
        [],
        [],
    )

    assert (
        result["root_cause"]
        is None
    )

    assert (
        result["score"]
        == 0
    )


def test_insufficient_evidence_returns_uncertain_result():

    assessor = RootCauseAssessor()

    hypotheses = [
        {
            "hypothesis":
                "Resource saturation may be contributing to application degradation."
        }
    ]

    result = assessor.assess(
        hypotheses,
        [],
    )

    assert (
        result["root_cause"]
        == hypotheses[0]["hypothesis"]
    )

    assert (
        result["score"]
        == 0
    )

    assert (
        "uncertain"
        in result["assessment"].lower()
    )