"""
Tests for HypothesisEngine.
"""

from src.agent.hypothesis_engine import (
    HypothesisEngine,
)


def test_resource_saturation_hypothesis():

    engine = HypothesisEngine()

    findings = [
        "High CPU utilization detected: 92.4%.",
        "High memory utilization detected: 81.7%.",
        "Application status is unhealthy.",
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 1

    hypothesis = hypotheses[0]

    assert (
        hypothesis["hypothesis"]
        == (
            "Resource saturation may be "
            "contributing to application "
            "degradation."
        )
    )

    assert hypothesis["confidence"] == "medium"

    assert hypothesis["requires_validation"] is True

    assert len(
        hypothesis["supporting_findings"]
    ) == 3


def test_degraded_instance_hypothesis():

    engine = HypothesisEngine()

    findings = [
        "Instance health is degraded.",
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 1

    assert (
        hypotheses[0]["hypothesis"]
        == (
            "The degraded instance health "
            "may be contributing to the "
            "observed incident."
        )
    )


def test_network_hypothesis():

    engine = HypothesisEngine()

    findings = [
        "Network status is unavailable.",
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 1

    assert (
        hypotheses[0]["hypothesis"]
        == (
            "A network-related problem may "
            "be contributing to the incident."
        )
    )


def test_application_error_hypothesis():

    engine = HypothesisEngine()

    findings = [
        (
            "Application errors detected: "
            "['Database connection timeout']"
        ),
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 1

    assert (
        hypotheses[0]["hypothesis"]
        == (
            "Application errors may be "
            "contributing to the observed "
            "service degradation."
        )
    )


def test_deployment_failure_hypothesis():

    engine = HypothesisEngine()

    findings = [
        (
            "A recent deployment failed "
            "and may be related to the incident."
        ),
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 1

    assert (
        hypotheses[0]["hypothesis"]
        == (
            "A recent failed deployment may "
            "be related to the incident."
        )
    )


def test_multiple_hypotheses_can_be_generated():

    engine = HypothesisEngine()

    findings = [
        "High CPU utilization detected: 92.4%.",
        "Application status is unhealthy.",
        "Instance health is degraded.",
        "Network status is unavailable.",
        (
            "Application errors detected: "
            "['HTTP 500 error']"
        ),
        (
            "A recent deployment failed "
            "and may be related to the incident."
        ),
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 5


def test_no_findings_generate_no_hypotheses():

    engine = HypothesisEngine()

    hypotheses = engine.generate([])

    assert hypotheses == []


def test_unrelated_findings_generate_no_hypotheses():

    engine = HypothesisEngine()

    findings = [
        "Some unrelated observation.",
        "Another unrelated observation.",
    ]

    hypotheses = engine.generate(findings)

    assert hypotheses == []