"""
Tool Executor

Maps LLM tool calls to the actual Python functions.
"""

import json

from src.tools.cloud_tools import (
    get_instance_health,
    get_application_logs,
    get_recent_deployments,
)


# --------------------------------------------------
# Tool Registry
# --------------------------------------------------

TOOL_REGISTRY = {
    "get_instance_health": get_instance_health,
    "get_application_logs": get_application_logs,
    "get_recent_deployments": get_recent_deployments,
}


# --------------------------------------------------
# Tool Execution
# --------------------------------------------------

def execute_tool(
    tool_name: str,
    arguments: dict,
):
    """
    Execute a tool requested by the LLM.

    Args:
        tool_name:
            Name of the requested tool.

        arguments:
            Arguments supplied by the LLM.

    Returns:
        Result returned by the selected tool.
    """

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    tool = TOOL_REGISTRY[tool_name]

    return tool(**arguments)


# --------------------------------------------------
# Tool Argument Parsing
# --------------------------------------------------

def parse_tool_arguments(arguments):
    """
    Convert tool arguments into a Python dictionary.

    Depending on the Cohere SDK response, arguments
    may already be a dictionary or may be returned
    as a JSON string.
    """

    if isinstance(arguments, dict):
        return arguments

    return json.loads(arguments)