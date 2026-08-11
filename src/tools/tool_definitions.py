"""
Tool definitions exposed to the LLM.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_instance_health",
            "description": (
                "Get the health and infrastructure metrics "
                "for a cloud instance. Use this tool when "
                "you need to investigate the health of a "
                "specific instance."
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
    }
]