"""
AI Cloud Operations Agent

Phase 2:
Multi-step tool-calling agent using Cohere.

The agent can:
1. Understand the user's question.
2. Select one or more tools.
3. Execute the requested tools.
4. Send tool results back to the LLM.
5. Allow the LLM to request additional tools.
6. Continue until the LLM produces a final answer.
7. Stop after a maximum number of iterations.

This implementation intentionally avoids LangGraph
so that the underlying agent loop is understood first.
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
    Multi-step AI Cloud Operations Agent.

    The agent uses an LLM to dynamically select and
    execute cloud operations tools.
    """

    # Maximum number of LLM/tool interaction cycles.
    #
    # This prevents the agent from entering an
    # infinite tool-calling loop.
    MAX_ITERATIONS = 5

    def __init__(self):
        self.llm = CohereClient()

    def run(self, question: str):
        """
        Process a user question using the LLM
        and available tools.

        The agent continues calling tools until:

        1. The LLM returns a final answer, or
        2. MAX_ITERATIONS is reached.
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
                    "information is required. "
                    "When investigating an incident, use "
                    "additional tools when the available "
                    "evidence is not sufficient to answer "
                    "the user's question confidently."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # --------------------------------------------------
        # Multi-step Agent Loop
        # --------------------------------------------------

        for iteration in range(1, self.MAX_ITERATIONS + 1):

            print(
                f"\n[Agent Iteration] {iteration}"
            )

            # --------------------------------------------------
            # Ask the LLM what to do next
            # --------------------------------------------------

            response = self.llm.chat(
                messages=messages,
                tools=TOOLS,
            )

            # --------------------------------------------------
            # Check whether the LLM wants to call a tool
            # --------------------------------------------------

            tool_calls = response.message.tool_calls

            # --------------------------------------------------
            # No tool call means the LLM has enough information
            # and is ready to provide the final answer.
            # --------------------------------------------------

            if not tool_calls:

                return response.message.content[0].text

            # --------------------------------------------------
            # Add the assistant's tool-call message to the
            # conversation history.
            # --------------------------------------------------

            messages.append(response.message)

            # --------------------------------------------------
            # Execute all tools requested by the LLM.
            # --------------------------------------------------

            for tool_call in tool_calls:

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

                # --------------------------------------------------
                # Execute the actual Python tool.
                # --------------------------------------------------

                result = execute_tool(
                    tool_name,
                    arguments,
                )

                print(
                    f"[Tool Result] "
                    f"{json.dumps(result)}"
                )

                # --------------------------------------------------
                # Convert the tool result into a Cohere-supported
                # tool-result content block.
                # --------------------------------------------------

                tool_content = [
                    {
                        "type": "document",
                        "document": {
                            "data": json.dumps(result)
                        },
                    }
                ]

                # --------------------------------------------------
                # Add the tool result to the conversation.
                #
                # The next LLM iteration will receive:
                #
                # User question
                # Assistant tool call
                # Tool result
                #
                # and can decide whether another tool is required.
                # --------------------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content,
                    }
                )

        # --------------------------------------------------
        # Safety Stop
        #
        # If the agent reaches MAX_ITERATIONS without
        # producing a final answer, stop the loop instead
        # of continuing indefinitely.
        # --------------------------------------------------

        return (
            "I was unable to complete the investigation "
            "within the allowed number of tool-calling "
            "iterations. Please try the question again "
            "with more specific information."
        )