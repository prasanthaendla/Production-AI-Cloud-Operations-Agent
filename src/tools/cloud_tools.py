"""
Cloud Operations Tools

Phase 1:
Mock cloud monitoring tools used to teach
LLM tool calling and agent workflows.

These tools simulate cloud infrastructure data.
No real AWS resources are accessed.
"""

from typing import Dict, Any


def get_instance_health(instance_id: str) -> Dict[str, Any]:
    """
    Return the simulated health information for a cloud instance.

    Args:
        instance_id: Cloud instance identifier.

    Returns:
        Dictionary containing instance health information.
    """

    mock_instances = {
        "i-demo-001": {
            "instance_id": "i-demo-001",
            "status": "running",
            "health": "degraded",
            "cpu_utilization": 92.4,
            "memory_utilization": 81.7,
            "network_status": "normal",
            "application_status": "unhealthy",
        },
        "i-demo-002": {
            "instance_id": "i-demo-002",
            "status": "running",
            "health": "healthy",
            "cpu_utilization": 34.2,
            "memory_utilization": 48.5,
            "network_status": "normal",
            "application_status": "healthy",
        },
    }

    return mock_instances.get(
        instance_id,
        {
            "instance_id": instance_id,
            "status": "unknown",
            "health": "unknown",
            "message": "Instance not found in mock environment.",
        },
    )


def get_application_logs(instance_id: str) -> Dict[str, Any]:
    """
    Return simulated application logs for a cloud instance.

    Args:
        instance_id: Cloud instance identifier.

    Returns:
        Dictionary containing application log information.
    """

    mock_logs = {
        "i-demo-001": {
            "instance_id": "i-demo-001",
            "log_count": 4,
            "logs": [
                {
                    "timestamp": "2026-08-11T17:42:10",
                    "level": "ERROR",
                    "message": (
                        "Database connection timeout "
                        "while processing request."
                    ),
                },
                {
                    "timestamp": "2026-08-11T17:42:15",
                    "level": "ERROR",
                    "message": (
                        "Request processing exceeded "
                        "30 seconds."
                    ),
                },
                {
                    "timestamp": "2026-08-11T17:42:21",
                    "level": "WARN",
                    "message": (
                        "Connection pool utilization "
                        "reached 95%."
                    ),
                },
                {
                    "timestamp": "2026-08-11T17:42:25",
                    "level": "ERROR",
                    "message": (
                        "HTTP 500 error rate increased "
                        "above normal threshold."
                    ),
                },
            ],
        },
        "i-demo-002": {
            "instance_id": "i-demo-002",
            "log_count": 1,
            "logs": [
                {
                    "timestamp": "2026-08-11T17:40:10",
                    "level": "INFO",
                    "message": (
                        "Application running normally."
                    ),
                }
            ],
        },
    }

    return mock_logs.get(
        instance_id,
        {
            "instance_id": instance_id,
            "log_count": 0,
            "logs": [],
            "message": (
                "No application logs found "
                "in mock environment."
            ),
        },
    )


def get_recent_deployments(instance_id: str) -> Dict[str, Any]:
    """
    Return simulated recent deployment information
    for a cloud instance.

    Args:
        instance_id: Cloud instance identifier.

    Returns:
        Dictionary containing recent deployment information.
    """

    mock_deployments = {
        "i-demo-001": {
            "instance_id": "i-demo-001",
            "deployment_count": 2,
            "deployments": [
                {
                    "deployment_id": "deploy-184",
                    "timestamp": "2026-08-11T17:35:00",
                    "version": "v2.8.1",
                    "status": "SUCCESS",
                    "deployed_by": "release-pipeline",
                },
                {
                    "deployment_id": "deploy-179",
                    "timestamp": "2026-08-08T10:20:00",
                    "version": "v2.7.9",
                    "status": "SUCCESS",
                    "deployed_by": "release-pipeline",
                },
            ],
        },
        "i-demo-002": {
            "instance_id": "i-demo-002",
            "deployment_count": 1,
            "deployments": [
                {
                    "deployment_id": "deploy-181",
                    "timestamp": "2026-08-10T14:15:00",
                    "version": "v2.8.0",
                    "status": "SUCCESS",
                    "deployed_by": "release-pipeline",
                }
            ],
        },
    }

    return mock_deployments.get(
        instance_id,
        {
            "instance_id": instance_id,
            "deployment_count": 0,
            "deployments": [],
            "message": (
                "No deployment information found "
                "in mock environment."
            ),
        },
    )