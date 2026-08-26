"""
Tool Executor

Maps LLM tool calls to the actual Python functions.

Stage 18.5:
- Validates requested tools.
- Validates tool arguments.
- Provides controlled argument parsing.
- Keeps actual tool execution failures visible to
  the LangGraph orchestration layer.
"""

from __future__ import annotations

import json
from typing import Any, Dict


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
    arguments: Dict[str, Any],
) -> Any:
    """
    Execute a tool requested by the LLM.

    Args:
        tool_name:
            Name of the requested tool.

        arguments:
            Arguments supplied by the LLM.

    Returns:
        Result returned by the selected tool.

    Raises:
        ValueError:
            If the tool is unknown or arguments are invalid.

        TypeError:
            If the supplied arguments are not a dictionary.

        Exception:
            Any exception raised by the actual tool is
            intentionally allowed to propagate to the
            LangGraph orchestration layer.

    Design:
        The executor does not hide runtime tool failures.
        LangGraph is responsible for deciding how the
        investigation should respond to a failed tool.
    """

    if not tool_name:
        raise ValueError(
            "Tool name cannot be empty."
        )

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    if not isinstance(arguments, dict):
        raise TypeError(
            "Tool arguments must be provided "
            "as a dictionary."
        )

    tool = TOOL_REGISTRY[tool_name]

    return tool(**arguments)


# --------------------------------------------------
# Tool Argument Parsing
# --------------------------------------------------

def parse_tool_arguments(
    arguments: Any,
) -> Dict[str, Any]:
    """
    Convert tool arguments into a Python dictionary.

    Depending on the Cohere SDK response, arguments
    may already be a dictionary or may be returned
    as a JSON string.

    Args:
        arguments:
            Tool arguments returned by the LLM.

    Returns:
        Parsed dictionary of tool arguments.

    Raises:
        ValueError:
            If the arguments are empty, malformed JSON,
            or the parsed value is not a dictionary.

        TypeError:
            If the supplied argument type is unsupported.
    """

    # --------------------------------------------------
    # Already parsed dictionary
    # --------------------------------------------------

    if isinstance(arguments, dict):

        return arguments

    # --------------------------------------------------
    # JSON string
    # --------------------------------------------------

    if isinstance(arguments, str):

        if not arguments.strip():

            raise ValueError(
                "Tool arguments cannot be empty."
            )

        try:

            parsed_arguments = json.loads(
                arguments
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                "Tool arguments contain invalid JSON."
            ) from exc

        if not isinstance(
            parsed_arguments,
            dict,
        ):

            raise ValueError(
                "Tool arguments JSON must represent "
                "an object/dictionary."
            )

        return parsed_arguments

    # --------------------------------------------------
    # Unsupported type
    # --------------------------------------------------

    raise TypeError(
        "Tool arguments must be either a dictionary "
        "or a JSON string."
    )


# --------------------------------------------------
# Tool Registry Helpers
# --------------------------------------------------

def is_tool_supported(
    tool_name: str,
) -> bool:
    """
    Check whether a tool is registered.

    This helper allows the orchestration layer to
    validate a tool before execution without directly
    accessing TOOL_REGISTRY.
    """

    return tool_name in TOOL_REGISTRY


def get_registered_tools() -> list[str]:
    """
    Return the names of all registered tools.
    """

    return list(
        TOOL_REGISTRY.keys()
    )