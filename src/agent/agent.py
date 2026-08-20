"""
AI Cloud Operations Agent

Phase 7:
Multi-step tool-calling agent with semantic scope
guardrails and capability-aware routing.

The agent can:

1. Determine whether a request belongs to the
   Cloud Operations domain using semantic similarity.
2. Reject unrelated requests before the agent loop.
3. Determine whether the request matches a capability
   currently supported by the agent.
4. Reject cloud-related requests that require
   unsupported capabilities.
5. Select one or more tools.
6. Execute requested tools.
7. Send tool results back to the LLM.
8. Allow the LLM to request additional tools.
9. Continue until the LLM produces a final answer.
10. Track investigation evidence.
11. Stop after a maximum number of iterations.

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

from src.agent.capability_router import (
    CapabilityRouter,
)


class CloudOperationsAgent:
    """
    Multi-step AI Cloud Operations Agent with:

    - semantic scope validation
    - capability-aware routing
    - tool calling
    - investigation evidence tracking
    """

    # Maximum number of LLM/tool interaction cycles.
    MAX_ITERATIONS = 5

    def __init__(self):
        """
        Initialize the Cloud Operations Agent.
        """

        self.llm = CohereClient()

        # --------------------------------------------------
        # Semantic scope classifier
        # --------------------------------------------------

        self.scope_classifier = (
            SemanticScopeClassifier()
        )

        # --------------------------------------------------
        # Capability router
        # --------------------------------------------------

        self.capability_router = (
            CapabilityRouter()
        )

    def run(self, question: str):
        """
        Process a user question.

        Processing flow:

        User Question
              ↓
        Semantic Scope Guardrail
              ↓
        Capability Router
              ↓
        Agent Loop
              ↓
        Tool Execution
              ↓
        Evidence
              ↓
        Final Answer
        """

        # ==================================================
        # Step 1: Semantic Scope Guardrail
        # ==================================================

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

        # ==================================================
        # Step 2: Capability Routing
        # ==================================================

        capability = (
            self.capability_router.route(
                question
            )
        )

        print(
            "\n[Capability Router]"
        )

        print(
            f"Supported: "
            f"{capability['is_supported']}"
        )

        print(
            f"Capability: "
            f"{capability['capability']}"
        )

        print(
            f"Confidence: "
            f"{capability['confidence']}"
        )

        print(
            f"Margin: "
            f"{capability['margin']}"
        )

        # --------------------------------------------------
        # Reject unsupported capabilities
        # --------------------------------------------------

        if not capability[
            "is_supported"
        ]:

            print(
                "\n[Capability Router] "
                "Request cannot be handled by "
                "the currently available capabilities."
            )

            return (
                "This is related to cloud, but I don't "
                "currently have the capability or tools "
                "required to answer this question."
            )

        # --------------------------------------------------
        # Capability accepted
        # --------------------------------------------------

        print(
            "\n[Capability Router] "
            "Using capability: "
            f"{capability['capability']}"
        )

        # ==================================================
        # Step 3: Conversation History
        # ==================================================

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

                    "When investigating an incident, use "
                    "additional tools when the available "
                    "evidence is not sufficient to answer "
                    "the user's question confidently. "

                    "Only answer the user's question using "
                    "available knowledge and tool results. "

                    "Do not claim to have performed an "
                    "operation unless an available tool "
                    "actually performed that operation."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # ==================================================
        # Investigation Evidence
        # ==================================================

        investigation_evidence = []

        # ==================================================
        # Step 4: Multi-step Agent Loop
        # ==================================================

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
                    "Evidence items collected: "
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
                # Record investigation evidence
                # --------------------------------------------------

                evidence_item = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
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
                    "Evidence Count: "
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

        # ==================================================
        # Step 5: Safety Stop
        # ==================================================

        print(
            "\n[Investigation Evidence]"
        )

        print(
            "Evidence items collected: "
            f"{len(investigation_evidence)}"
        )

        return (
            "I was unable to complete the investigation "
            "within the allowed number of tool-calling "
            "iterations. Please try the question again "
            "with more specific information."
        )