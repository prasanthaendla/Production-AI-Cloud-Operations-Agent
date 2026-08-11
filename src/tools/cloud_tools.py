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