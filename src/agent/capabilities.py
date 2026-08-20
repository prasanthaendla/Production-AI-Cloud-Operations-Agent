"""
Agent Capability Registry

Defines the capabilities currently supported by the
AI Cloud Operations Agent.

The capability registry describes what the agent can
actually investigate with its current tools.
"""


CAPABILITIES = {
    "instance_health": {
        "description": (
            "Investigate the health of a specific cloud "
            "instance using infrastructure health metrics "
            "such as CPU utilization, memory utilization, "
            "network status, instance status, application "
            "status, degraded health, unhealthy instances, "
            "and infrastructure health."
        ),
        "tools": [
            "get_instance_health",
        ],
    },
    "application_logs": {
        "description": (
            "Investigate recent application logs for a "
            "specific cloud instance, including application "
            "errors, warnings, HTTP errors, database "
            "connection errors, connection timeouts, "
            "application failures, and log messages."
        ),
        "tools": [
            "get_application_logs",
        ],
    },
    "deployments": {
        "description": (
            "Investigate cloud application deployments, "
            "failed deployments, successful deployments, "
            "recent deployments, release versions, release "
            "pipelines, CI/CD deployments, deployment "
            "status, and whether a recent deployment may "
            "be related to an incident."
        ),
        "tools": [
            "get_recent_deployments",
        ],
    },
    "incident_investigation": {
        "description": (
            "Investigate a production cloud incident by "
            "correlating instance health metrics, "
            "application logs, errors, application health, "
            "recent deployments, release information, "
            "performance problems, and other available "
            "operational evidence to identify likely "
            "causes."
        ),
        "tools": [
            "get_instance_health",
            "get_application_logs",
            "get_recent_deployments",
        ],
    },
}


def get_capabilities():
    """
    Return all capabilities supported by the agent.
    """

    return CAPABILITIES


def get_capability_names():
    """
    Return the names of all supported capabilities.
    """

    return list(
        CAPABILITIES.keys()
    )


def get_capability_descriptions():
    """
    Return capability names and descriptions.
    """

    return {
        name: capability["description"]
        for name, capability in CAPABILITIES.items()
    }