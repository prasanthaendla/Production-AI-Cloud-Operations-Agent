"""
Tests for ConfidenceEngine.
"""

from src.agent.confidence_engine import (
    ConfidenceEngine,
)


def test_confirmed_root_cause_has_high_confidence():

    engine = ConfidenceEngine()

    assessment = {
        "root_cause": (
            "Resource saturation"
        ),
        "score": 4,
        "supporting_evidence": [
            "High CPU",
            "High memory",
        ],
        "contradicting_evidence": [],
    }

    result = engine.evaluate(
        assessment
    )

    assert (
        result["confidence_level"]
        == "HIGH"
    )

    assert (
        result["confidence_score"]
        == 0.90
    )


def test_supporting_and_contradicting_evidence_has_medium_confidence():

    engine = ConfidenceEngine()

    assessment = {
        "root_cause": (
            "Resource saturation"
        ),
        "score": 2,
        "supporting_evidence": [
            "High CPU",
        ],
        "contradicting_evidence": [
            "Network normal",
        ],
    }

    result = engine.evaluate(
        assessment
    )

    assert (
        result["confidence_level"]
        == "MEDIUM"
    )


def test_single_supporting_evidence_has_medium_confidence():

    engine = ConfidenceEngine()

    assessment = {
        "root_cause": (
            "Application failure"
        ),
        "score": 1,
        "supporting_evidence": [
            "Application errors",
        ],
        "contradicting_evidence": [],
    }

    result = engine.evaluate(
        assessment
    )

    assert (
        result["confidence_level"]
        == "MEDIUM"
    )


def test_no_supporting_evidence_has_low_confidence():

    engine = ConfidenceEngine()

    assessment = {
        "root_cause": (
            "Unknown failure"
        ),
        "score": 0,
        "supporting_evidence": [],
        "contradicting_evidence": [],
    }

    result = engine.evaluate(
        assessment
    )

    assert (
        result["confidence_level"]
        == "LOW"
    )


def test_uncertainty_mentions_further_investigation():

    engine = ConfidenceEngine()

    assessment = {
        "root_cause": (
            "Unknown failure"
        ),
        "score": 0,
        "supporting_evidence": [],
        "contradicting_evidence": [
            "Normal health",
        ],
    }

    result = engine.evaluate(
        assessment
    )

    assert (
        "further investigation"
        in result["uncertainty"].lower()
    )


def test_evaluate_all_processes_multiple_assessments():

    engine = ConfidenceEngine()

    assessments = [
        {
            "root_cause": "Resource saturation",
            "score": 4,
            "supporting_evidence": [
                "High CPU",
                "High memory",
            ],
            "contradicting_evidence": [],
        },
        {
            "root_cause": "Network failure",
            "score": 0,
            "supporting_evidence": [],
            "contradicting_evidence": [],
        },
    ]

    results = engine.evaluate_all(
        assessments
    )

    assert len(results) == 2

    assert (
        results[0]["confidence_level"]
        == "HIGH"
    )

    assert (
        results[1]["confidence_level"]
        == "LOW"
    )