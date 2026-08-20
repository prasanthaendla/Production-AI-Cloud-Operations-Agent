"""
Tests for the capability router.
"""

from src.agent.capability_router import CapabilityRouter


def test_instance_health_capability():
    router = CapabilityRouter()

    result = router.route(
        "Why is instance i-demo-001 unhealthy?"
    )

    assert result["is_supported"] is True
    assert result["capability"] == "instance_health"


def test_application_logs_capability():
    router = CapabilityRouter()

    result = router.route(
        "Why are there errors in my application logs?"
    )

    assert result["is_supported"] is True
    assert result["capability"] == "application_logs"


def test_deployment_capability():
    router = CapabilityRouter()

    result = router.route(
        "Why did my deployment fail?"
    )

    assert result["is_supported"] is True
    assert result["capability"] == "deployments"


def test_unsupported_cloud_capability():
    router = CapabilityRouter()

    result = router.route(
        "What are the top cloud providers?"
    )

    assert result["is_supported"] is False
    assert result["capability"] is None