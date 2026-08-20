"""
AI Cloud Operations Agent

Phase 8:
Multi-step tool-calling agent with:

- semantic scope guardrails
- capability-aware routing
- structured investigation state
- tool calling
- evidence tracking

The implementation intentionally avoids LangGraph
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

from src.agent.investigation_state import (
    InvestigationState,
)


class CloudOperationsAgent:
    """
    Multi-step AI Cloud Operations Agent.

    Responsibilities:

    1. Semantic scope validation
    2. Capability routing
    3. Investigation state management
    4. Tool selection
    5. Tool execution
    6. Evidence collection
    7. Final investigation response
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

    def run(
        self,
        question: str,
    ):
        """
        Process a user question.

        Processing flow:

        User Question
              ↓
        Semantic Scope Guardrail
              ↓
        Capability Router
              ↓
        Investigation State
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

        print(
            "\n[Capability Router] "
            "Using capability: "
            f"{capability['capability']}"
        )

        # ==================================================
        # Step 3: Create Investigation State
        # ==================================================

        investigation = InvestigationState(
            question=question,
            capability=(
                capability["capability"]
            ),
        )

        print(
            "\n[Investigation State]"
        )

        print(
            f"Capability: "
            f"{investigation.capability}"
        )

        # ==================================================
        # Step 4: Conversation History
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
        # Step 5: Multi-step Agent Loop
        # ==================================================

        for iteration in range(
            1,
            self.MAX_ITERATIONS + 1,
        ):

            # --------------------------------------------------
            # Record iteration in investigation state
            # --------------------------------------------------

            investigation.record_iteration(
                iteration
            )

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
                    "\n[Investigation State]"
                )

                print(
                    f"Iterations: "
                    f"{investigation.iterations}"
                )

                print(
                    f"Tool Calls: "
                    f"{len(investigation.tool_calls)}"
                )

                print(
                    f"Evidence Items: "
                    f"{len(investigation.evidence)}"
                )

                print(
                    f"Findings: "
                    f"{len(investigation.findings)}"
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
                # Record requested tool call
                # --------------------------------------------------

                investigation.record_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
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

                investigation.record_evidence(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                )

                print(
                    "\n[Evidence Recorded]"
                )

                print(
                    f"Tool: {tool_name}"
                )

                print(
                    "Evidence Count: "
                    f"{len(investigation.evidence)}"
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
        # Step 6: Safety Stop
        # ==================================================

        print(
            "\n[Investigation State]"
        )

        print(
            f"Iterations: "
            f"{investigation.iterations}"
        )

        print(
            f"Tool Calls: "
            f"{len(investigation.tool_calls)}"
        )

        print(
            f"Evidence Items: "
            f"{len(investigation.evidence)}"
        )

        return (
            "I was unable to complete the investigation "
            "within the allowed number of tool-calling "
            "iterations. Please try the question again "
            "with more specific information."
        )