import pytest

from src.guardrails.semantic_classifier import (
    SemanticScopeClassifier,
)


@pytest.fixture(scope="module")
def classifier():
    return SemanticScopeClassifier()


# ============================================================
# CLOUD OPERATIONS — SHOULD BE ACCEPTED
# ============================================================


@pytest.mark.parametrize(
    "question",
    [
        "What is an EC2 instance?",
        "Why is my EC2 CPU utilization high?",
        "Why is my application unhealthy?",
        "What errors are occurring in my application logs?",
        "Why did my deployment fail?",
        "How do I troubleshoot high latency?",
        "What is a VPC?",
        "What is a subnet in AWS?",
        "What is a route table?",
        "Why is my cloud server unavailable?",
        "How can I monitor CPU and memory utilization?",
        "Why is my Kubernetes pod restarting?",
        "How do I troubleshoot a Kubernetes deployment?",
        "What is AWS networking?",
        "Why is my cloud application timing out?",
        "How do I investigate a production incident?",
        "What is infrastructure monitoring?",
        "How can I troubleshoot a cloud outage?",
        "Why is my load balancer unhealthy?",
        "How do I investigate an application failure?",
    ],
)
def test_cloud_operations_questions_are_allowed(
    classifier,
    question,
):

    result = classifier.classify(question)

    assert result["is_cloud_operations"] is True, (
        f"Expected CLOUD_OPERATIONS but received "
        f"{result} for question: {question}"
    )


# ============================================================
# OUT OF SCOPE — SHOULD BE REJECTED
# ============================================================


@pytest.mark.parametrize(
    "question",
    [
        "What is my name?",
        "What is India's capital city?",
        "What is Agentic AI?",
        "Explain Python programming.",
        "How do I prepare for an exam?",
        "Tell me a joke.",
        "Write me a poem.",
        "What is the stock market?",
        "What is the weather today?",
        "Who is the president of India?",
        "What is machine learning?",
        "Explain generative AI.",
        "What is a neural network?",
        "How do I learn Java?",
        "What are Oracle 21c new features?",
    ],
)
def test_non_cloud_questions_are_rejected(
    classifier,
    question,
):

    result = classifier.classify(question)

    assert result["is_cloud_operations"] is False, (
        f"Expected OUT_OF_SCOPE but received "
        f"{result} for question: {question}"
    )