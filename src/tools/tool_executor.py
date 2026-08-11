"""
Tool Executor

Maps LLM tool calls to actual Python functions.
"""

import json

from src.tools.cloud_tools import (
    get_instance_health,
)


TOOL_REGISTRY = {
    "get_instance_health": get_instance_health,
}


def execute_tool(
    tool_name: str,
    arguments: dict,
):
    """
    Execute a tool requested by the LLM.

    Args:
        tool_name: Name of the requested tool.
        arguments: Arguments supplied by the LLM.

    Returns:
        Tool execution result.
    """

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    tool = TOOL_REGISTRY[tool_name]

    return tool(**arguments)


def parse_tool_arguments(arguments):
    """
    Convert tool arguments into a Python dictionary.

    Cohere may return arguments as a JSON string
    depending on the SDK response.
    """

    if isinstance(arguments, dict):
        return arguments

    return json.loads(arguments)