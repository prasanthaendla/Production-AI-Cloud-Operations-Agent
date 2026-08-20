
"""
AI Cloud Operations Agent

Phase 6:
Evidence-based multi-step cloud incident investigation.

The agent can:

1. Determine whether a request belongs to the
   Cloud Operations domain using semantic similarity.
2. Reject unrelated requests before the agent loop.
3. Select one or more tools.
4. Execute requested tools.
5. Store tool results as investigation evidence.
6. Send tool results back to the LLM.
7. Allow the LLM to request additional tools.
8. Continue until the LLM produces a final answer.
9. Stop after a maximum number of iterations.

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
from src.guardrails.semantic_classifier import (
    SemanticScopeClassifier,
)


class CloudOperationsAgent:
    """
    Multi-step AI Cloud Operations Agent with
    semantic scope validation and investigation
    evidence tracking.
    """

    # Maximum number of LLM/tool interaction cycles.
    MAX_ITERATIONS = 5

    def __init__(self):
        """
        Initialize the Cloud Operations Agent.
        """

        self.llm = CohereClient()

        self.scope_classifier = (
            SemanticScopeClassifier()
        )

    def run(self, question: str):
        """
        Process a user question.

        The request first passes through the semantic
        scope classifier.

        Valid requests enter the multi-step agent loop.

        Tool results are stored as investigation evidence
        so that the LLM can reason over the complete
        investigation.
        """

        # --------------------------------------------------
        # Step 1: Semantic Scope Guardrail
        # --------------------------------------------------

        classification = (
            self.scope_classifier.classify(
                question
            )
        )

        print(
            "\n[Semantic Guardrail]"
        )

        print(
            f"Category: "
            f"{classification['category']}"
        )

        print(
            f"Confidence: "
            f"{classification['confidence']}"
        )

        if classification.get(
            "matched_domain"
        ):
            print(
                f"Matched Domain: "
                f"{classification['matched_domain']}"
            )

        # --------------------------------------------------
        # Reject out-of-scope requests
        # --------------------------------------------------

        if not classification[
            "is_cloud_operations"
        ]:

            print(
                "\n[Semantic Guardrail] "
                "Request rejected as out-of-scope."
            )

            return (
                "I can only help with questions "
                "related to cloud infrastructure "
                "and cloud operations."
            )

        # --------------------------------------------------
        # Accept cloud-related request
        # --------------------------------------------------

        print(
            "\n[Semantic Guardrail] "
            "Request accepted."
        )

        # --------------------------------------------------
        # Investigation Evidence
        # --------------------------------------------------
        #
        # Every successful tool execution is stored here.
        #
        # Example:
        #
        # {
        #     "tool": "get_instance_health",
        #     "arguments": {
        #         "instance_id": "i-demo-001"
        #     },
        #     "result": {...}
        # }
        #
        # This gives the investigation a persistent
        # evidence trail during the current request.
        # --------------------------------------------------

        investigation_evidence = []

        # --------------------------------------------------
        # Conversation History
        # --------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Cloud Operations Agent. "

                    "Your responsibility is to investigate "
                    "cloud infrastructure and application "
                    "operations using the available tools. "

                    "Supported areas include cloud "
                    "infrastructure, AWS, Azure, GCP, "
                    "Kubernetes, Docker, networking, "
                    "monitoring, application health, "
                    "logs, deployments, incidents, "
                    "performance and troubleshooting. "

                    "Do not invent infrastructure metrics. "

                    "Use tools when factual infrastructure "
                    "information is required. "

                    "When investigating an incident, "
                    "analyze the available evidence before "
                    "producing the final answer. "

                    "If the current evidence is insufficient "
                    "to answer the question confidently, "
                    "request another appropriate tool. "

                    "Use the results of previous tools when "
                    "deciding what additional evidence is "
                    "required. "

                    "Do not claim that one event caused "
                    "another unless the available evidence "
                    "supports that conclusion. "

                    "Clearly distinguish observed facts "
                    "from analysis or likely causes. "

                    "Only answer the user's question using "
                    "available knowledge and tool results."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # --------------------------------------------------
        # Step 2: Multi-step Agent Loop
        # --------------------------------------------------

        for iteration in range(
            1,
            self.MAX_ITERATIONS + 1,
        ):

            print(
                f"\n[Agent Iteration] "
                f"{iteration}"
            )

            # --------------------------------------------------
            # Ask the LLM what to do next
            # --------------------------------------------------

            response = self.llm.chat(
                messages=messages,
                tools=TOOLS,
            )

            # --------------------------------------------------
            # Check whether the LLM requested tools
            # --------------------------------------------------

            tool_calls = (
                response.message.tool_calls
            )

            # --------------------------------------------------
            # No tool call means the LLM has enough
            # information to produce the final answer.
            # --------------------------------------------------

            if not tool_calls:

                print(
                    "\n[Investigation Evidence]"
                )

                print(
                    f"Evidence items collected: "
                    f"{len(investigation_evidence)}"
                )

                return (
                    response.message
                    .content[0]
                    .text
                )

            # --------------------------------------------------
            # Add assistant tool-call message
            # --------------------------------------------------

            messages.append(
                response.message
            )

            # --------------------------------------------------
            # Execute requested tools
            # --------------------------------------------------

            for tool_call in tool_calls:

                tool_name = (
                    tool_call.function.name
                )

                arguments = (
                    parse_tool_arguments(
                        tool_call.function.arguments
                    )
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
                # Execute actual Python tool
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
                # Store investigation evidence
                # --------------------------------------------------

                evidence_item = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "iteration": iteration,
                }

                investigation_evidence.append(
                    evidence_item
                )

                print(
                    "\n[Evidence Recorded]"
                )

                print(
                    f"Tool: {tool_name}"
                )

                print(
                    f"Evidence Count: "
                    f"{len(investigation_evidence)}"
                )

                # --------------------------------------------------
                # Convert tool result into
                # Cohere-compatible tool content.
                # --------------------------------------------------

                tool_content = [
                    {
                        "type": "document",
                        "document": {
                            "data": json.dumps(
                                result
                            )
                        },
                    }
                ]

                # --------------------------------------------------
                # Add tool result to conversation history.
                #
                # The next LLM iteration receives the
                # tool result and can decide whether
                # additional evidence is required.
                # --------------------------------------------------

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call.id
                        ),
                        "content": tool_content,
                    }
                )

        # --------------------------------------------------
        # Step 3: Safety Stop
        # --------------------------------------------------

        print(
            "\n[Investigation Evidence]"
        )

        print(
            f"Evidence items collected: "
            f"{len(investigation_evidence)}"
        )

        return (
            "I was unable to complete the investigation "
            "within the allowed number of tool-calling "
            "iterations. Please try the question again "
            "with more specific information."
        )