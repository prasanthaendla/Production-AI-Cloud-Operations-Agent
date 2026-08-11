"""
AI Cloud Operations Agent

Phase 1:
Simple tool-calling agent using Cohere.

This implementation intentionally avoids
LangGraph so that the underlying agent
loop is understood first.
"""

import json

from src.llm.cohere_client import CohereClient
from src.tools.tool_definitions import TOOLS
from src.tools.tool_executor import (
    execute_tool,
    parse_tool_arguments,
)


class CloudOperationsAgent:
    """
    Simple tool-calling agent.
    """

    def __init__(self):
        self.llm = CohereClient()

    def run(self, question: str):
        """
        Process a user question using the LLM
        and available tools.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Cloud Operations Agent. "
                    "Investigate cloud infrastructure problems "
                    "using the available tools. "
                    "Do not invent infrastructure metrics. "
                    "Use tools when factual infrastructure "
                    "information is required."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # --------------------------------------------------
        # Step 1: Ask the LLM whether a tool is required
        # --------------------------------------------------

        response = self.llm.chat(
            messages=messages,
            tools=TOOLS,
        )

        # --------------------------------------------------
        # Step 2: If no tool is required, return the answer
        # --------------------------------------------------

        if not response.message.tool_calls:

            return response.message.content[0].text

        # --------------------------------------------------
        # Step 3: Add the assistant tool-call message
        # --------------------------------------------------

        messages.append(response.message)

        # --------------------------------------------------
        # Step 4: Execute each requested tool
        # --------------------------------------------------

        for tool_call in response.message.tool_calls:

            tool_name = tool_call.function.name

            arguments = parse_tool_arguments(
                tool_call.function.arguments
            )

            print(
                f"\n[Agent Tool Call] "
                f"{tool_name}"
            )

            print(
                f"[Arguments] "
                f"{json.dumps(arguments)}"
            )

            # Execute the actual Python function
            result = execute_tool(
                tool_name,
                arguments,
            )

            print(
                f"[Tool Result] "
                f"{json.dumps(result)}"
            )

            # --------------------------------------------------
            # Send the tool result back to Cohere
            #
            # Cohere V2 expects tool results as supported
            # content blocks. We use a document block so the
            # model receives structured tool information.
            # --------------------------------------------------

            tool_content = [
                {
                    "type": "document",
                    "document": {
                        "data": json.dumps(result)
                    },
                }
            ]

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )

        # --------------------------------------------------
        # Step 5: Ask the LLM to generate the final answer
        # using the tool results
        # --------------------------------------------------

        final_response = self.llm.chat(
            messages=messages,
            tools=TOOLS,
        )

        return final_response.message.content[0].text