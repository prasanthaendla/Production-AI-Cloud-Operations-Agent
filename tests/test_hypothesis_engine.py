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

    assert (
        hypothesis["recommended_evidence"]
        == [
            "application_logs",
            "deployments",
        ]
    )


def test_degraded_instance_hypothesis():

    engine = HypothesisEngine()

    findings = [
        "Instance health is degraded.",
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 1

    hypothesis = hypotheses[0]

    assert (
        hypothesis["hypothesis"]
        == (
            "The degraded instance health "
            "may be contributing to the "
            "observed incident."
        )
    )

    assert hypothesis["confidence"] == "medium"

    assert hypothesis["requires_validation"] is True

    assert (
        hypothesis["recommended_evidence"]
        == [
            "application_logs",
            "deployments",
        ]
    )


def test_network_hypothesis():

    engine = HypothesisEngine()

    findings = [
        "Network status is unavailable.",
    ]

    hypotheses = engine.generate(findings)

    assert len(hypotheses) == 1

    hypothesis = hypotheses[0]

    assert (
        hypothesis["hypothesis"]
        == (
            "A network-related problem may "
            "be contributing to the incident."
        )
    )

    assert hypothesis["confidence"] == "medium"

    assert hypothesis["requires_validation"] is True

    assert (
        hypothesis["recommended_evidence"]
        == [
            "application_logs",
        ]
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

    hypothesis = hypotheses[0]

    assert (
        hypothesis["hypothesis"]
        == (
            "Application errors may be "
            "contributing to the observed "
            "service degradation."
        )
    )

    assert hypothesis["confidence"] == "medium"

    assert hypothesis["requires_validation"] is True

    assert (
        hypothesis["recommended_evidence"]
        == [
            "deployments",
        ]
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

    hypothesis = hypotheses[0]

    assert (
        hypothesis["hypothesis"]
        == (
            "A recent failed deployment may "
            "be related to the incident."
        )
    )

    assert hypothesis["confidence"] == "medium"

    assert hypothesis["requires_validation"] is True

    assert (
        hypothesis["recommended_evidence"]
        == [
            "application_logs",
        ]
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


def test_resource_hypothesis_recommends_additional_evidence():

    engine = HypothesisEngine()

    findings = [
        "High CPU utilization detected: 92.4%.",
        "High memory utilization detected: 81.7%.",
        "Application status is unhealthy.",
    ]

    hypotheses = engine.generate(findings)

    evidence = engine.get_recommended_evidence(
        hypotheses
    )

    assert "application_logs" in evidence

    assert "deployments" in evidence


def test_recommended_evidence_is_unique():

    engine = HypothesisEngine()

    hypotheses = [
        {
            "requires_validation": True,
            "recommended_evidence": [
                "application_logs",
                "deployments",
            ],
        },
        {
            "requires_validation": True,
            "recommended_evidence": [
                "application_logs",
            ],
        },
    ]

    evidence = engine.get_recommended_evidence(
        hypotheses
    )

    assert evidence == [
        "application_logs",
        "deployments",
    ]


def test_hypothesis_without_validation_has_no_recommended_evidence():

    engine = HypothesisEngine()

    hypotheses = [
        {
            "requires_validation": False,
            "recommended_evidence": [
                "application_logs",
            ],
        }
    ]

    evidence = engine.get_recommended_evidence(
        hypotheses
    )

    assert evidence == []