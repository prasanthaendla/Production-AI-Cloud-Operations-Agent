"""
Tests for the semantic scope classifier.
"""

import pytest

from src.guardrails.semantic_classifier import (
    SemanticScopeClassifier,
)


@pytest.fixture
def classifier():
    return SemanticScopeClassifier()


def test_ec2_question_is_cloud_operations(
    classifier,
):
    result = classifier.classify(
        "What is an EC2 instance?"
    )

    assert result["is_cloud_operations"] is True
    assert result["category"] == "CLOUD_OPERATIONS"


def test_gcp_question_is_cloud_operations(
    classifier,
):
    result = classifier.classify(
        "What is GCP?"
    )

    assert result["is_cloud_operations"] is True
    assert result["category"] == "CLOUD_OPERATIONS"


def test_kubernetes_question_is_cloud_operations(
    classifier,
):
    result = classifier.classify(
        "Why is my Kubernetes application unhealthy?"
    )

    assert result["is_cloud_operations"] is True
    assert result["category"] == "CLOUD_OPERATIONS"


def test_incident_question_is_cloud_operations(
    classifier,
):
    result = classifier.classify(
        "Why is my production server responding slowly?"
    )

    assert result["is_cloud_operations"] is True
    assert result["category"] == "CLOUD_OPERATIONS"


def test_aws_networking_question_is_cloud_operations(
    classifier,
):
    result = classifier.classify(
        "What is a VPC in AWS?"
    )

    assert result["is_cloud_operations"] is True
    assert result["category"] == "CLOUD_OPERATIONS"


def test_application_logs_question_is_cloud_operations(
    classifier,
):
    result = classifier.classify(
        "What errors are appearing in my application logs?"
    )

    assert result["is_cloud_operations"] is True
    assert result["category"] == "CLOUD_OPERATIONS"


def test_personal_question_is_out_of_scope(
    classifier,
):
    result = classifier.classify(
        "Do you know my name?"
    )

    assert result["is_cloud_operations"] is False
    assert result["category"] == "OUT_OF_SCOPE"


def test_general_question_is_out_of_scope(
    classifier,
):
    result = classifier.classify(
        "What is my favorite movie?"
    )

    assert result["is_cloud_operations"] is False
    assert result["category"] == "OUT_OF_SCOPE"


def test_agentic_ai_question_is_out_of_scope(
    classifier,
):
    result = classifier.classify(
        "What is Agentic AI?"
    )

    assert result["is_cloud_operations"] is False
    assert result["category"] == "OUT_OF_SCOPE"