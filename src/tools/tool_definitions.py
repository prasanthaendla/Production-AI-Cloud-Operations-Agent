"""
Tool definitions exposed to the LLM.

These definitions describe the tools available
to the AI Cloud Operations Agent.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_instance_health",
            "description": (
                "Get the health and infrastructure metrics "
                "for a cloud instance. Use this tool when "
                "you need to investigate CPU, memory, network, "
                "application status, or overall health of "
                "a specific instance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": (
                            "The cloud instance identifier, "
                            "for example i-demo-001."
                        ),
                    }
                },
                "required": [
                    "instance_id"
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_application_logs",
            "description": (
                "Retrieve recent application logs for a "
                "cloud instance. Use this tool when you "
                "need to investigate application errors, "
                "warnings, HTTP errors, database connection "
                "problems, timeouts, or other application "
                "failures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": (
                            "The cloud instance identifier, "
                            "for example i-demo-001."
                        ),
                    }
                },
                "required": [
                    "instance_id"
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_deployments",
            "description": (
                "Get recent deployment information for a "
                "cloud instance. Use this tool when you "
                "need to investigate whether a recent "
                "deployment may be related to an incident "
                "or application failure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instance_id": {
                        "type": "string",
                        "description": (
                            "The cloud instance identifier, "
                            "for example i-demo-001."
                        ),
                    }
                },
                "required": [
                    "instance_id"
                ],
            },
        },
    },
]