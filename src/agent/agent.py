"""
AI Cloud Operations Agent

Multi-step cloud operations investigation agent with:

- Semantic scope guardrails
- Capability-aware routing
- Investigation state
- Tool calling
- Evidence tracking
- Deterministic investigation analysis
- Hypothesis generation
- Evidence-driven additional tool selection

Root cause assessment will be integrated only after
this investigation loop is verified end-to-end.
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

from src.agent.investigation_analyzer import (
    InvestigationAnalyzer,
)

from src.agent.hypothesis_engine import (
    HypothesisEngine,
)


class CloudOperationsAgent:
    """
    Multi-step AI Cloud Operations Agent.

    Investigation flow:

        Question
            ↓
        Semantic Guardrail
            ↓
        Capability Router
            ↓
        Investigation State
            ↓
        Initial Tool
            ↓
        Evidence
            ↓
        Investigation Analyzer
            ↓
        Findings
            ↓
        Hypothesis Engine
            ↓
        Evidence Recommendations
            ↓
        Additional Tools
            ↓
        Additional Evidence
            ↓
        Final Answer
    """

    MAX_ITERATIONS = 5

    def __init__(self):

        # --------------------------------------------------
        # LLM
        # --------------------------------------------------

        self.llm = CohereClient()

        # --------------------------------------------------
        # Guardrail
        # --------------------------------------------------

        self.scope_classifier = (
            SemanticScopeClassifier()
        )

        # --------------------------------------------------
        # Capability Router
        # --------------------------------------------------

        self.capability_router = (
            CapabilityRouter()
        )

        # --------------------------------------------------
        # Investigation Analyzer
        # --------------------------------------------------

        self.investigation_analyzer = (
            InvestigationAnalyzer()
        )

        # --------------------------------------------------
        # Hypothesis Engine
        # --------------------------------------------------

        self.hypothesis_engine = (
            HypothesisEngine()
        )

    # ======================================================
    # MAIN AGENT
    # ======================================================

    def run(
        self,
        question: str,
    ):

        # ==================================================
        # 1. Semantic Guardrail
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
        # 2. Capability Router
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

        if not capability[
            "is_supported"
        ]:

            print(
                "\n[Capability Router] "
                "Request cannot be handled."
            )

            return (
                "This is related to cloud operations, "
                "but I don't currently have the "
                "capability or tools required to "
                "answer this question."
            )

        print(
            "\n[Capability Router] "
            "Using capability: "
            f"{capability['capability']}"
        )

        # ==================================================
        # 3. Investigation State
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
        # 4. Conversation
        # ==================================================

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Cloud Operations Agent. "
                    "Investigate cloud infrastructure and "
                    "application incidents using the "
                    "available tools. "
                    "Use tools when infrastructure evidence "
                    "is required. "
                    "Do not invent metrics or tool results. "
                    "When the available evidence is "
                    "insufficient, continue investigating "
                    "using appropriate available tools. "
                    "Do not stop after the first tool call "
                    "when additional evidence is required."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        # ==================================================
        # 5. Investigation Loop
        # ==================================================

        for iteration in range(
            1,
            self.MAX_ITERATIONS + 1,
        ):

            investigation.record_iteration(
                iteration
            )

            print(
                f"\n[Agent Iteration] "
                f"{iteration}"
            )

            # --------------------------------------------------
            # Ask LLM
            # --------------------------------------------------

            response = self.llm.chat(
                messages=messages,
                tools=TOOLS,
            )

            tool_calls = (
                response.message.tool_calls
            )

            # ==================================================
            # If LLM wants to finish, check whether
            # investigation still requires validation.
            # ==================================================

            if not tool_calls:

                # ----------------------------------------------
                # Check current hypotheses
                # ----------------------------------------------

                hypotheses = getattr(
                    investigation,
                    "hypotheses",
                    [],
                )

                recommended_evidence = (
                    self.hypothesis_engine
                    .get_recommended_evidence(
                        hypotheses
                    )
                )

                # ----------------------------------------------
                # Determine tools already executed
                # ----------------------------------------------

                executed_tools = {
                    call.get("tool")
                    for call in (
                        investigation.tool_calls
                        if isinstance(
                            investigation.tool_calls,
                            list,
                        )
                        else []
                    )
                }

                evidence_tool_mapping = {
                    "application_logs":
                        "get_application_logs",

                    "deployments":
                        "get_recent_deployments",
                }

                missing_validation_tools = []

                for evidence_type in (
                    recommended_evidence
                ):

                    tool_name = (
                        evidence_tool_mapping.get(
                            evidence_type
                        )
                    )

                    if (
                        tool_name
                        and tool_name
                        not in executed_tools
                    ):

                        missing_validation_tools.append(
                            tool_name
                        )

                if missing_validation_tools:

                    print(
                        "\n[Evidence Validation]"
                    )

                    print(
                        "LLM attempted to finish "
                        "while validation evidence "
                        "is still missing."
                    )

                    print(
                        "Requesting additional "
                        "investigation."
                    )

                    # ------------------------------------------
                    # Tell LLM that validation is required
                    # ------------------------------------------

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Additional investigation "
                                "is required before answering. "
                                "The following validation tools "
                                "have not yet been executed: "
                                + ", ".join(
                                    missing_validation_tools
                                )
                                + ". "
                                "Use the appropriate tools "
                                "to collect this evidence."
                            ),
                        }
                    )

                    continue

                # ----------------------------------------------
                # Investigation complete
                # ----------------------------------------------

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

                print(
                    f"Hypotheses: "
                    f"{len(hypotheses)}"
                )

                return (
                    response.message
                    .content[0]
                    .text
                )

            # ==================================================
            # 6. Process Tool Calls
            # ==================================================

            messages.append(
                response.message
            )

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

                # ----------------------------------------------
                # Record tool call
                # ----------------------------------------------

                investigation.record_tool_call(
                    tool_name=tool_name,
                    arguments=arguments,
                )

                # ----------------------------------------------
                # Execute tool
                # ----------------------------------------------

                result = execute_tool(
                    tool_name,
                    arguments,
                )

                print(
                    f"[Tool Result] "
                    f"{json.dumps(result)}"
                )

                # ----------------------------------------------
                # Record evidence
                # ----------------------------------------------

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

                # ==================================================
                # 7. Analyze Evidence
                # ==================================================

                findings = (
                    self.investigation_analyzer
                    .analyze(
                        investigation.evidence
                    )
                )

                # ----------------------------------------------
                # Replace findings with current deterministic
                # analysis
                # ----------------------------------------------

                investigation.findings = []

                for finding in findings:

                    investigation.add_finding(
                        finding
                    )

                print(
                    "\n[Investigation Analysis]"
                )

                print(
                    f"Findings Generated: "
                    f"{len(investigation.findings)}"
                )

                for finding in (
                    investigation.findings
                ):

                    print(
                        f"- {finding}"
                    )

                # ==================================================
                # 8. Generate Hypotheses
                # ==================================================

                hypotheses = (
                    self.hypothesis_engine.generate(
                        investigation.findings
                    )
                )

                # ----------------------------------------------
                # Store hypotheses
                # ----------------------------------------------

                if hasattr(
                    investigation,
                    "hypotheses",
                ):

                    investigation.hypotheses = (
                        hypotheses
                    )

                print(
                    "\n[Hypothesis Analysis]"
                )

                print(
                    f"Hypotheses Generated: "
                    f"{len(hypotheses)}"
                )

                for hypothesis in hypotheses:

                    if isinstance(
                        hypothesis,
                        dict,
                    ):

                        print(
                            "- "
                            + str(
                                hypothesis.get(
                                    "hypothesis",
                                    hypothesis,
                                )
                            )
                        )

                    else:

                        print(
                            f"- {hypothesis}"
                        )

                # ==================================================
                # 9. Evidence-Driven Investigation
                # ==================================================

                recommended_evidence = (
                    self.hypothesis_engine
                    .get_recommended_evidence(
                        hypotheses
                    )
                )

                evidence_tool_mapping = {
                    "application_logs":
                        "get_application_logs",

                    "deployments":
                        "get_recent_deployments",
                }

                executed_tools = set()

                for call in (
                    investigation.tool_calls
                ):

                    if isinstance(
                        call,
                        dict,
                    ):

                        tool_name_from_call = (
                            call.get("tool")
                        )

                        if tool_name_from_call:

                            executed_tools.add(
                                tool_name_from_call
                            )

                missing_validation_tools = []

                for evidence_type in (
                    recommended_evidence
                ):

                    tool_name_for_evidence = (
                        evidence_tool_mapping.get(
                            evidence_type
                        )
                    )

                    if (
                        tool_name_for_evidence
                        and tool_name_for_evidence
                        not in executed_tools
                    ):

                        missing_validation_tools.append(
                            tool_name_for_evidence
                        )

                if missing_validation_tools:

                    print(
                        "\n[Evidence-Driven Investigation]"
                    )

                    print(
                        "Additional validation tools "
                        "required: "
                        f"{missing_validation_tools}"
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Continue the investigation. "
                                "The evidence currently "
                                "collected is not sufficient "
                                "to validate the hypotheses. "
                                "Use these additional tools "
                                "where appropriate: "
                                + ", ".join(
                                    missing_validation_tools
                                )
                            ),
                        }
                    )

                else:

                    print(
                        "\n[Evidence-Driven Investigation]"
                    )

                    print(
                        "No additional validation "
                        "tools required."
                    )

                # ==================================================
                # 10. Add Tool Result + Investigation Context
                # ==================================================

                investigation_context = {
                    "findings": (
                        investigation.findings
                    ),
                    "hypotheses": hypotheses,
                    "recommended_evidence": (
                        recommended_evidence
                    ),
                }

                tool_content = [
                    {
                        "type": "document",
                        "document": {
                            "data": json.dumps(
                                {
                                    "tool_result": result,
                                    "investigation_analysis": (
                                        investigation_context
                                    ),
                                }
                            )
                        },
                    }
                ]

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
        # 11. Maximum Iteration Safety Stop
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

        print(
            f"Findings: "
            f"{len(investigation.findings)}"
        )

        hypotheses = getattr(
            investigation,
            "hypotheses",
            [],
        )

        print(
            f"Hypotheses: "
            f"{len(hypotheses)}"
        )

        return (
            "I was unable to complete the cloud "
            "operations investigation within the "
            "maximum investigation iterations."
        )