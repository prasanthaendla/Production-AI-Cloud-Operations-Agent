"""
LangGraph orchestration for the AI Cloud Operations Agent.

Stage 18 - Production Hardening

Architecture:

    Question
       |
       v
    Boundary Validation
       |
       v
    Guardrail
       |
       v
    Capability Router
       |
       v
    Investigation State
       |
       v
    Initial Investigation
       |
       v
    Tool Execution
       |
       v
    Evidence Analysis
       |
       v
    Hypothesis Analysis
       |
       v
    Evidence Decision
       |
       +----------------------+
       |                      |
       | More evidence        | Sufficient evidence
       v                      v
    Tool Execution        Root Cause
                              |
                              v
                         Confidence
                              |
                              v
                             RAG
                              |
                              v
                           Memory
                              |
                              v
                            Answer

Important:
- LangGraph controls orchestration and state transitions.
- Existing project components provide the intelligence.
- CloudOperationsAgent.run() is NOT called from this workflow.
- Existing guardrails, router, analyzer, hypothesis engine,
  root-cause assessor, confidence engine, RAG and memory are reused.
- Public API validation happens before LangGraph execution.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

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

from src.agent.root_cause_assessor import (
    RootCauseAssessor,
)

from src.agent.confidence_engine import (
    ConfidenceEngine,
)

from src.knowledge.knowledge_retriever import (
    KnowledgeRetriever,
)

from src.memory.incident_memory import (
    IncidentMemory,
)

from src.observability.tracer import InvestigationTracer


class CloudInvestigationGraphState(TypedDict, total=False):
    """
    State carried through the complete LangGraph investigation.
    """

    question: str
    capability: str
    capability_confidence: float

    iteration: int
    max_iterations: int

    status: str
    error: str
    next_action: str

    messages: List[Any]

    investigation: Any

    tool_calls: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    findings: List[str]
    hypotheses: List[Any]

    recommended_evidence: List[str]
    missing_validation_tools: List[str]

    knowledge: List[Dict[str, Any]]
    historical_incidents: List[Dict[str, Any]]

    root_cause_assessment: Dict[str, Any]
    confidence_assessment: Dict[str, Any]

    root_cause: str
    root_cause_score: int

    confidence_level: str
    confidence_score: float

    final_answer: str

    pending_tool_calls: List[Dict[str, Any]]

    trace_summary: Dict[str, Any]
    trace_events: List[Dict[str, Any]]

    tool_failures: List[Dict[str, Any]]


class CloudOperationsLangGraph:
    """
    Real node-level LangGraph orchestration.

    This class intentionally does NOT call:

        CloudOperationsAgent.run()

    because doing so would hide the investigation inside one large
    operation and defeat the purpose of LangGraph.

    Instead, LangGraph coordinates the same existing components that
    CloudOperationsAgent already uses.
    """

    DEFAULT_MAX_ITERATIONS = 5

    MIN_MAX_ITERATIONS = 1
    MAX_MAX_ITERATIONS = 10

    def __init__(
        self,
        llm: CohereClient | None = None,
        scope_classifier: SemanticScopeClassifier | None = None,
        capability_router: CapabilityRouter | None = None,
        investigation_analyzer: InvestigationAnalyzer | None = None,
        hypothesis_engine: HypothesisEngine | None = None,
        root_cause_assessor: RootCauseAssessor | None = None,
        confidence_engine: ConfidenceEngine | None = None,
        knowledge_retriever: KnowledgeRetriever | None = None,
        incident_memory: IncidentMemory | None = None,
    ) -> None:

        self.llm = llm or CohereClient()

        self.scope_classifier = (
            scope_classifier
            or SemanticScopeClassifier()
        )

        self.capability_router = (
            capability_router
            or CapabilityRouter()
        )

        self.investigation_analyzer = (
            investigation_analyzer
            or InvestigationAnalyzer()
        )

        self.hypothesis_engine = (
            hypothesis_engine
            or HypothesisEngine()
        )

        self.root_cause_assessor = (
            root_cause_assessor
            or RootCauseAssessor()
        )

        self.confidence_engine = (
            confidence_engine
            or ConfidenceEngine()
        )

        self.knowledge_retriever = (
            knowledge_retriever
            or KnowledgeRetriever()
        )

        self.incident_memory = (
            incident_memory
            or IncidentMemory()
        )

        self.tracer = InvestigationTracer()

        self.graph = self._build_graph()

    # ==============================================================
    # GRAPH CONSTRUCTION
    # ==============================================================

    def _build_graph(self):

        workflow = StateGraph(
            CloudInvestigationGraphState
        )

        workflow.add_node(
            "initialize",
            self._trace_node(
                "initialize",
                self._initialize,
            ),
        )

        workflow.add_node(
            "investigate",
            self._trace_node(
                "investigate",
                self._investigate,
            ),
        )

        workflow.add_node(
            "execute_tools",
            self._trace_node(
                "execute_tools",
                self._execute_tools,
            ),
        )

        workflow.add_node(
            "analyze_evidence",
            self._trace_node(
                "analyze_evidence",
                self._analyze_evidence,
            ),
        )

        workflow.add_node(
            "decide_next",
            self._trace_node(
                "decide_next",
                self._decide_next,
            ),
        )

        workflow.add_node(
            "assess_root_cause",
            self._trace_node(
                "assess_root_cause",
                self._assess_root_cause,
            ),
        )

        workflow.add_node(
            "assess_confidence",
            self._trace_node(
                "assess_confidence",
                self._assess_confidence,
            ),
        )

        workflow.add_node(
            "generate_answer",
            self._trace_node(
                "generate_answer",
                self._generate_answer,
            ),
        )

        workflow.add_node(
            "save_memory",
            self._trace_node(
                "save_memory",
                self._save_memory,
            ),
        )

        workflow.add_edge(
            START,
            "initialize",
        )

        workflow.add_conditional_edges(
            "initialize",
            self._route_after_initialize,
            {
                "investigate": "investigate",
                "complete": "generate_answer",
            },
        )

        workflow.add_conditional_edges(
            "investigate",
            self._route_after_investigation,
            {
                "execute_tools": "execute_tools",
                "analyze": "analyze_evidence",
                "complete": "generate_answer",
            },
        )

        workflow.add_edge(
            "execute_tools",
            "analyze_evidence",
        )

        workflow.add_edge(
            "analyze_evidence",
            "decide_next",
        )

        workflow.add_conditional_edges(
            "decide_next",
            self._route_after_decision,
            {
                "investigate": "investigate",
                "root_cause": "assess_root_cause",
                "complete": "generate_answer",
            },
        )

        workflow.add_edge(
            "assess_root_cause",
            "assess_confidence",
        )

        workflow.add_edge(
            "assess_confidence",
            "generate_answer",
        )

        workflow.add_edge(
            "generate_answer",
            "save_memory",
        )

        workflow.add_edge(
            "save_memory",
            END,
        )

        return workflow.compile()

    def _trace_node(
        self,
        node_name: str,
        node_function,
    ):
        """
        Wrap a LangGraph node with observability tracing.
        """

        def traced_node(
            state: CloudInvestigationGraphState,
        ) -> Dict[str, Any]:

            start_time = self.tracer.node_started(
                node_name,
                iteration=state.get("iteration"),
            )

            try:

                result = node_function(
                    state
                )

                self.tracer.node_completed(
                    node_name,
                    start_time,
                    status=result.get(
                        "status",
                        "completed",
                    ),
                )

                return result

            except Exception as exc:

                self.tracer.error(
                    node_name,
                    exc,
                )

                self.tracer.node_completed(
                    node_name,
                    start_time,
                    status="failed",
                )

                raise

        return traced_node

    # ==============================================================
    # INITIALIZE
    # ==============================================================

    def _initialize(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        question = state.get(
            "question",
            "",
        ).strip()

        if not question:

            return {
                "status": "failed",
                "error": (
                    "Investigation question "
                    "cannot be empty."
                ),
                "next_action": "complete",
            }

        print(
            "\n[LangGraph Node] initialize"
        )

        # ----------------------------------------------------------
        # Semantic Guardrail
        # ----------------------------------------------------------

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

            return {
                "status": "failed",
                "error": (
                    "I can only help with questions "
                    "related to cloud infrastructure "
                    "and cloud operations."
                ),
                "next_action": "complete",
            }

        print(
            "\n[Semantic Guardrail] "
            "Request accepted."
        )

        # ----------------------------------------------------------
        # Capability Router
        # ----------------------------------------------------------

        capability_result = (
            self.capability_router.route(
                question
            )
        )

        print(
            "\n[Capability Router]"
        )

        print(
            f"Supported: "
            f"{capability_result['is_supported']}"
        )

        print(
            f"Capability: "
            f"{capability_result['capability']}"
        )

        print(
            f"Confidence: "
            f"{capability_result['confidence']}"
        )

        print(
            f"Margin: "
            f"{capability_result['margin']}"
        )

        if not capability_result[
            "is_supported"
        ]:

            return {
                "status": "failed",
                "error": (
                    "This is related to cloud operations, "
                    "but I don't currently have the "
                    "capability or tools required to "
                    "answer this question."
                ),
                "next_action": "complete",
            }

        capability = (
            capability_result["capability"]
        )

        print(
            "\n[Capability Router] "
            f"Using capability: {capability}"
        )

        # ----------------------------------------------------------
        # Investigation State
        # ----------------------------------------------------------

        investigation = InvestigationState(
            question=question,
            capability=capability,
        )

        print(
            "\n[Investigation State]"
        )

        print(
            f"Capability: "
            f"{investigation.capability}"
        )

        # ----------------------------------------------------------
        # Initial Operational Knowledge
        # ----------------------------------------------------------

        knowledge_results = (
            self.knowledge_retriever
            .retrieve_for_investigation(
                query=question,
                findings=investigation.findings,
                evidence=investigation.evidence,
                top_k=3,
            )
        )

        print(
            "\n[Operational Knowledge]"
        )

        print(
            "Relevant documents: "
            f"{len(knowledge_results)}"
        )

        for knowledge in knowledge_results:

            print(
                f"- {knowledge['source']} "
                f"(score={knowledge['score']})"
            )

        # ----------------------------------------------------------
        # Historical Memory
        # ----------------------------------------------------------

        historical_incidents = (
            self.incident_memory.retrieve_similar(
                question=question,
                findings=investigation.findings,
                capability=investigation.capability,
                top_k=3,
            )
        )

        print(
            "\n[Historical Incident Memory]"
        )

        print(
            "Similar historical incidents: "
            f"{len(historical_incidents)}"
        )

        for incident in historical_incidents:

            print(
                f"- {incident['incident_id']} "
                f"(score={incident['similarity_score']})"
            )

        # ----------------------------------------------------------
        # LLM conversation
        # ----------------------------------------------------------

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
                    "When evidence is insufficient, continue "
                    "investigating. "
                    "Do not stop after the first tool call "
                    "when additional evidence is required. "
                    "Use actual cloud evidence as the primary "
                    "basis for root cause analysis. "
                    "Operational knowledge provides guidance "
                    "only. "
                    "Historical incidents provide context only."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        return {
            "question": question,
            "capability": capability,
            "capability_confidence": float(
                capability_result.get(
                    "confidence",
                    0.0,
                )
            ),
            "investigation": investigation,
            "messages": messages,
            "knowledge": knowledge_results,
            "historical_incidents": historical_incidents,
            "iteration": 0,
            "max_iterations": state.get(
                "max_iterations",
                self.DEFAULT_MAX_ITERATIONS,
            ),
            "tool_calls": [],
            "evidence": [],
            "findings": [],
            "hypotheses": [],
            "recommended_evidence": [],
            "missing_validation_tools": [],
            "pending_tool_calls": [],
            "tool_failures": [],
            "status": "initialized",
            "next_action": "investigate",
        }

    # ==============================================================
    # INITIAL ROUTING
    # ==============================================================

    @staticmethod
    def _route_after_initialize(
        state: CloudInvestigationGraphState,
    ) -> str:

        if state.get(
            "status"
        ) == "failed":

            return "complete"

        return "investigate"

    # ==============================================================
    # INVESTIGATE NODE
    # ==============================================================

    def _investigate(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        if state.get(
            "status"
        ) == "failed":

            return {
                "next_action": "complete",
            }

        iteration = (
            state.get(
                "iteration",
                0,
            )
            + 1
        )

        max_iterations = state.get(
            "max_iterations",
            self.DEFAULT_MAX_ITERATIONS,
        )

        if iteration > max_iterations:

            return {
                "status": "failed",
                "error": (
                    "Maximum investigation "
                    "iterations exceeded."
                ),
                "next_action": "complete",
            }

        investigation = state[
            "investigation"
        ]

        investigation.record_iteration(
            iteration
        )

        print(
            f"\n[LangGraph Node] investigate "
            f"(iteration {iteration})"
        )

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        response = self.llm.chat(
            messages=messages,
            tools=TOOLS,
        )

        tool_calls = (
            response.message.tool_calls
        )

        if not tool_calls:

            print(
                "\n[LLM Decision] "
                "No tool call requested."
            )

            return {
                "iteration": iteration,
                "status": "analysis_ready",
                "pending_tool_calls": [],
                "next_action": "analyze",
            }

        pending_tool_calls = []

        for tool_call in tool_calls:

            tool_name = (
                tool_call.function.name
            )

            arguments = (
                parse_tool_arguments(
                    tool_call.function.arguments
                )
            )

            pending_tool_calls.append(
                {
                    "id": tool_call.id,
                    "tool": tool_name,
                    "arguments": arguments,
                }
            )

        print(
            "\n[LangGraph Decision]"
        )

        print(
            "Tools selected: "
            f"{[call['tool'] for call in pending_tool_calls]}"
        )

        messages.append(
            response.message
        )

        return {
            "iteration": iteration,
            "messages": messages,
            "pending_tool_calls": pending_tool_calls,
            "status": "tools_selected",
            "next_action": "execute_tools",
        }

    # ==============================================================
    # INVESTIGATION ROUTING
    # ==============================================================

    @staticmethod
    def _route_after_investigation(
        state: CloudInvestigationGraphState,
    ) -> str:

        if state.get(
            "status"
        ) == "failed":

            return "complete"

        action = state.get(
            "next_action",
            "analyze",
        )

        if action == "execute_tools":
            return "execute_tools"

        if action == "analyze":
            return "analyze"

        return "complete"

    # ==============================================================
    # TOOL EXECUTION
    # ==============================================================

    def _execute_tools(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:
        """
        Execute pending tools without allowing one tool failure to
        terminate the complete investigation.

        Stage 18 production-hardening behavior:
        - Successful tools produce normal evidence.
        - Failed tools are recorded as structured evidence.
        - The tracer records the failure.
        - The LLM receives the failure as a tool result.
        - The workflow continues to evidence analysis.

        This prevents transient cloud/API/tool failures from causing
        the complete LangGraph invocation to crash.
        """

        investigation = state[
            "investigation"
        ]

        pending_tool_calls = state.get(
            "pending_tool_calls",
            [],
        )

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        all_tool_calls = list(
            state.get(
                "tool_calls",
                [],
            )
        )

        all_evidence = list(
            state.get(
                "evidence",
                [],
            )
        )

        tool_failures = list(
            state.get(
                "tool_failures",
                [],
            )
        )

        print(
            "\n[LangGraph Node] execute_tools"
        )

        for tool_call in pending_tool_calls:

            tool_name = tool_call[
                "tool"
            ]

            arguments = tool_call[
                "arguments"
            ]

            call_id = tool_call.get(
                "id",
                tool_name,
            )

            print(
                f"\n[Agent Tool Call] "
                f"{tool_name}"
            )

            print(
                f"[Arguments] "
                f"{json.dumps(arguments)}"
            )

            investigation.record_tool_call(
                tool_name=tool_name,
                arguments=arguments,
            )

            tool_start_time = self.tracer.tool_started(
                tool_name,
                arguments,
            )

            try:
                result = execute_tool(
                    tool_name,
                    arguments,
                )

                self.tracer.tool_completed(
                    tool_name,
                    tool_start_time,
                )

                print(
                    f"[Tool Result] "
                    f"{json.dumps(result)}"
                )

                investigation.record_evidence(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                )

                self.tracer.evidence_recorded(
                    tool_name,
                    len(all_evidence) + 1,
                )

                all_tool_calls.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "status": "success",
                    }
                )

                all_evidence.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result,
                        "status": "success",
                    }
                )

                tool_content = [
                    {
                        "type": "document",
                        "document": {
                            "data": json.dumps(
                                {
                                    "tool_result": result,
                                }
                            ),
                        },
                    }
                ]

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_content,
                    }
                )

            except Exception as exc:
                # --------------------------------------------------
                # Production-safe tool failure handling
                # --------------------------------------------------

                error_message = str(exc).strip() or (
                    exc.__class__.__name__
                )

                failure_result = {
                    "tool_error": True,
                    "tool": tool_name,
                    "error_type": exc.__class__.__name__,
                    "error": error_message,
                    "message": (
                        "Tool execution failed. "
                        "Do not invent the missing result."
                    ),
                }

                self.tracer.error(
                    f"tool:{tool_name}",
                    exc,
                )

                self.tracer.tool_completed(
                    tool_name,
                    tool_start_time,
                    status="failed",
                )

                print(
                    f"[Tool Error] {tool_name}: "
                    f"{error_message}"
                )

                failure_record = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "error_type": exc.__class__.__name__,
                    "error": error_message,
                }

                tool_failures.append(
                    failure_record
                )

                # Record the failure as evidence so downstream nodes
                # know exactly which evidence could not be collected.
                investigation.record_evidence(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=failure_result,
                )

                self.tracer.evidence_recorded(
                    tool_name,
                    len(all_evidence) + 1,
                )

                all_tool_calls.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": failure_result,
                        "status": "failed",
                    }
                )

                all_evidence.append(
                    {
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": failure_result,
                        "status": "failed",
                    }
                )

                tool_content = [
                    {
                        "type": "document",
                        "document": {
                            "data": json.dumps(
                                {
                                    "tool_result": failure_result,
                                }
                            ),
                        },
                    }
                ]

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_content,
                    }
                )

                # Continue with any remaining tools. A failure in one
                # tool must not prevent independent tools from running.
                continue

        print(
            "\n[Evidence Recorded]"
        )

        print(
            "Evidence Count: "
            f"{len(all_evidence)}"
        )

        if tool_failures:
            print(
                "Tool Failures: "
                f"{len(tool_failures)}"
            )

        return {
            "messages": messages,
            "tool_calls": all_tool_calls,
            "evidence": all_evidence,
            "tool_failures": tool_failures,
            "pending_tool_calls": [],
            "status": (
                "tool_execution_completed_with_errors"
                if tool_failures
                else "evidence_collected"
            ),
            "next_action": "analyze",
        }

    # ==============================================================
    # EVIDENCE ANALYSIS
    # ==============================================================

    def _analyze_evidence(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        investigation = state[
            "investigation"
        ]

        evidence = state.get(
            "evidence",
            [],
        )

        tool_failures = state.get(
            "tool_failures",
            [],
        )

        question = state[
            "question"
        ]

        print(
            "\n[LangGraph Node] "
            "analyze_evidence"
        )

        # ----------------------------------------------------------
        # Investigation Analyzer
        # ----------------------------------------------------------

        findings = (
            self.investigation_analyzer.analyze(
                evidence
            )
        )

        # Explicitly surface tool failures as findings. This prevents
        # downstream reasoning from treating missing evidence as proof
        # that the underlying condition is healthy.
        for failure in tool_failures:
            findings.append(
                "Tool execution failed for "
                f"{failure['tool']}: "
                f"{failure['error']}"
            )

        investigation.findings = []

        for finding in findings:

            investigation.add_finding(
                finding
            )

        print(
            "\n[Investigation Analysis]"
        )

        print(
            "Findings Generated: "
            f"{len(investigation.findings)}"
        )

        for finding in (
            investigation.findings
        ):

            print(
                f"- {finding}"
            )

        # ----------------------------------------------------------
        # Evidence-aware RAG
        # ----------------------------------------------------------

        knowledge_results = (
            self.knowledge_retriever
            .retrieve_for_investigation(
                query=question,
                findings=investigation.findings,
                evidence=evidence,
                top_k=3,
            )
        )

        print(
            "\n[Evidence-Aware Knowledge]"
        )

        print(
            "Relevant documents: "
            f"{len(knowledge_results)}"
        )

        for knowledge in knowledge_results:

            print(
                f"- {knowledge['source']} "
                f"(score={knowledge['score']})"
            )

        # ----------------------------------------------------------
        # Hypothesis Engine
        # ----------------------------------------------------------

        hypotheses = (
            self.hypothesis_engine.generate(
                investigation.findings
            )
        )

        investigation.hypotheses = (
            hypotheses
        )

        print(
            "\n[Hypothesis Analysis]"
        )

        print(
            "Hypotheses Generated: "
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

        executed_tools = {
            call.get("tool")
            for call in (
                investigation.tool_calls
            )
            if isinstance(
                call,
                dict,
            )
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

        print(
            "\n[Evidence-Driven Investigation]"
        )

        if missing_validation_tools:

            print(
                "Additional validation tools "
                "required: "
                f"{missing_validation_tools}"
            )

        else:

            print(
                "No additional validation "
                "tools required."
            )

        # ----------------------------------------------------------
        # Add investigation context to messages
        # ----------------------------------------------------------

        investigation_context = {
            "findings": (
                investigation.findings
            ),
            "hypotheses": hypotheses,
            "tool_failures": tool_failures,
            "recommended_evidence": (
                recommended_evidence
            ),
            "operational_knowledge": [
                {
                    "source": knowledge[
                        "source"
                    ],
                    "content": knowledge[
                        "content"
                    ],
                    "score": knowledge[
                        "score"
                    ],
                }
                for knowledge
                in knowledge_results
            ],
        }

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    "Investigation evidence and "
                    "analysis:\n"
                    + json.dumps(
                        investigation_context
                    )
                    + "\n\nContinue the investigation "
                    "if additional evidence is required. "
                    "Otherwise proceed toward root cause."
                ),
            }
        )

        return {
            "messages": messages,
            "findings": list(
                investigation.findings
            ),
            "hypotheses": hypotheses,
            "recommended_evidence": (
                recommended_evidence
            ),
            "missing_validation_tools": (
                missing_validation_tools
            ),
            "knowledge": knowledge_results,
            "status": "evidence_analyzed",
        }

    # ==============================================================
    # DECISION
    # ==============================================================

    def _decide_next(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        iteration = state.get(
            "iteration",
            0,
        )

        max_iterations = state.get(
            "max_iterations",
            self.DEFAULT_MAX_ITERATIONS,
        )

        missing_tools = state.get(
            "missing_validation_tools",
            [],
        )

        evidence = state.get(
            "evidence",
            [],
        )

        print(
            "\n[LangGraph Node] decide_next"
        )

        # ----------------------------------------------------------
        # Safety limit
        # ----------------------------------------------------------

        if iteration >= max_iterations:

            print(
                "Maximum iterations reached."
            )

            self.tracer.decision(
                decision="root_cause",
                reason=(
                    "Maximum investigation "
                    "iterations reached."
                ),
                iteration=iteration,
            )

            return {
                "next_action": "root_cause",
                "status": "root_cause_ready",
            }

        # ----------------------------------------------------------
        # More evidence required
        # ----------------------------------------------------------

        if missing_tools:

            print(
                "Decision: additional "
                "evidence required."
            )

            self.tracer.decision(
                decision="investigate",
                reason=(
                    "Additional validation tools "
                    "are required."
                ),
                iteration=iteration,
            )

            pending_tool_calls = []

            instance_id = (
                self._extract_instance_id(
                    state.get(
                        "question",
                        "",
                    ),
                    evidence,
                )
            )

            for tool_name in missing_tools:

                arguments = {}

                if instance_id:

                    arguments[
                        "instance_id"
                    ] = instance_id

                pending_tool_calls.append(
                    {
                        "id": (
                            f"langgraph-{iteration}-"
                            f"{tool_name}"
                        ),
                        "tool": tool_name,
                        "arguments": arguments,
                    }
                )

            return {
                "pending_tool_calls": (
                    pending_tool_calls
                ),
                "next_action": "investigate",
                "status": "more_evidence_required",
            }

        # ----------------------------------------------------------
        # Evidence sufficient
        # ----------------------------------------------------------

        print(
            "Decision: evidence sufficient."
        )

        self.tracer.decision(
            decision="root_cause",
            reason=(
                "Evidence is sufficient for "
                "root-cause assessment."
            ),
            iteration=iteration,
        )

        return {
            "next_action": "root_cause",
            "status": "root_cause_ready",
        }

    # ==============================================================
    # ROUTING AFTER DECISION
    # ==============================================================

    @staticmethod
    def _route_after_decision(
        state: CloudInvestigationGraphState,
    ) -> str:

        action = state.get(
            "next_action",
            "complete",
        )

        if action == "investigate":
            return "investigate"

        if action == "root_cause":
            return "root_cause"

        return "complete"

    # ==============================================================
    # ROOT CAUSE
    # ==============================================================

    def _assess_root_cause(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        investigation = state[
            "investigation"
        ]

        hypotheses = state.get(
            "hypotheses",
            [],
        )

        evidence = state.get(
            "evidence",
            [],
        )

        print(
            "\n[LangGraph Node] "
            "assess_root_cause"
        )

        root_cause_result = (
            self.root_cause_assessor.assess(
                hypotheses=hypotheses,
                evidence=evidence,
            )
        )

        root_cause = (
            root_cause_result.get(
                "root_cause",
                "",
            )
        )

        score = (
            root_cause_result.get(
                "score",
                0,
            )
        )

        print(
            "\n[Root Cause Assessment]"
        )

        print(
            f"Most Likely Root Cause: "
            f"{root_cause}"
        )

        print(
            f"Root Cause Score: "
            f"{score}"
        )

        print(
            root_cause_result.get(
                "assessment",
                "",
            )
        )

        investigation.root_cause_assessment = (
            root_cause_result
        )

        investigation.root_cause = (
            root_cause
        )

        return {
            "root_cause_assessment": (
                root_cause_result
            ),
            "root_cause": root_cause,
            "root_cause_score": score,
            "status": "root_cause_assessed",
        }

    # ==============================================================
    # CONFIDENCE
    # ==============================================================

    def _assess_confidence(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        investigation = state[
            "investigation"
        ]

        root_cause_result = state.get(
            "root_cause_assessment",
            {},
        )

        print(
            "\n[LangGraph Node] "
            "assess_confidence"
        )

        confidence_input = dict(
            root_cause_result
        )

        if (
            "supporting_evidence"
            not in confidence_input
        ):

            confidence_input[
                "supporting_evidence"
            ] = root_cause_result.get(
                "supporting_evidence",
                [],
            )

        if (
            "contradicting_evidence"
            not in confidence_input
        ):

            confidence_input[
                "contradicting_evidence"
            ] = root_cause_result.get(
                "contradicting_evidence",
                [],
            )

        confidence_result = (
            self.confidence_engine.evaluate(
                confidence_input
            )
        )

        investigation.confidence_assessment = (
            confidence_result
        )

        print(
            "\n[Confidence Assessment]"
        )

        print(
            "Confidence Level: "
            f"{confidence_result['confidence_level']}"
        )

        print(
            "Confidence Score: "
            f"{confidence_result['confidence_score']}"
        )

        print(
            "Supporting Evidence: "
            f"{confidence_result['supporting_evidence_count']}"
        )

        print(
            "Contradicting Evidence: "
            f"{confidence_result['contradicting_evidence_count']}"
        )

        print(
            "Uncertainty: "
            f"{confidence_result['uncertainty']}"
        )

        return {
            "confidence_assessment": (
                confidence_result
            ),
            "confidence_level": (
                confidence_result[
                    "confidence_level"
                ]
            ),
            "confidence_score": (
                confidence_result[
                    "confidence_score"
                ]
            ),
            "status": "confidence_assessed",
        }

    # ==============================================================
    # FINAL ANSWER
    # ==============================================================

    def _generate_answer(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        if state.get(
            "status"
        ) == "failed":

            return {
                "final_answer": state.get(
                    "error",
                    "Investigation failed.",
                ),
                "status": "failed",
            }

        question = state[
            "question"
        ]

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        root_cause = state.get(
            "root_cause_assessment",
            {},
        )

        confidence = state.get(
            "confidence_assessment",
            {},
        )

        knowledge = state.get(
            "knowledge",
            [],
        )

        historical = state.get(
            "historical_incidents",
            [],
        )

        tool_failures = state.get(
            "tool_failures",
            [],
        )

        evidence = state.get(
            "evidence",
            [],
        )

        findings = state.get(
            "findings",
            [],
        )

        hypotheses = state.get(
            "hypotheses",
            [],
        )

        print(
            "\n[LangGraph Node] "
            "generate_answer"
        )

        context = {
            "question": question,
            "findings": findings,
            "hypotheses": hypotheses,
            "evidence": evidence,
            "tool_failures": tool_failures,
            "root_cause_assessment": root_cause,
            "confidence_assessment": confidence,
            "operational_knowledge": knowledge,
            "historical_incidents": historical,
        }

        messages.append(
            {
                "role": "user",
                "content": (
                    "The investigation is complete. "
                    "Generate the final answer using "
                    "the following investigation context.\n\n"
                    + json.dumps(
                        context
                    )
                    + "\n\n"
                    "Actual cloud evidence must be "
                    "prioritized over runbooks and "
                    "historical incidents. "
                    "Clearly distinguish evidence from "
                    "inference. "
                    "Mention confidence and uncertainty. "
                    "If any tool failed, explicitly state which "
                    "evidence was unavailable and do not claim "
                    "that the missing evidence was negative or "
                    "successful."
                ),
            }
        )

        response = self.llm.chat(
            messages=messages,
            tools=TOOLS,
        )

        final_answer = (
            response.message
            .content[0]
            .text
        )

        print(
            "\n[LangGraph Final Answer Generated]"
        )

        return {
            "messages": messages,
            "final_answer": final_answer,
            "status": "answer_generated",
        }

    # ==============================================================
    # SAVE MEMORY
    # ==============================================================

    def _save_memory(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:

        if state.get(
            "status"
        ) == "failed":

            return {
                "status": "failed",
            }

        investigation = state[
            "investigation"
        ]

        print(
            "\n[LangGraph Node] save_memory"
        )

        saved_incident = (
            self.incident_memory.save_incident(
                question=state[
                    "question"
                ],
                capability=investigation.capability,
                findings=state.get(
                    "findings",
                    [],
                ),
                hypotheses=state.get(
                    "hypotheses",
                    [],
                ),
                root_cause_assessment=state.get(
                    "root_cause_assessment",
                    {},
                ),
                confidence_assessment=state.get(
                    "confidence_assessment",
                    {},
                ),
            )
        )

        print(
            "\n[Historical Incident Memory]"
        )

        print(
            "Incident saved: "
            f"{saved_incident['incident_id']}"
        )

        print(
            "Total historical incidents: "
            f"{self.incident_memory.count()}"
        )

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
            f"{len(investigation.hypotheses)}"
        )

        return {
            "status": "completed",
            "final_answer": state.get(
                "final_answer",
                "",
            ),
        }

    # ==============================================================
    # INSTANCE ID EXTRACTION
    # ==============================================================

    @staticmethod
    def _extract_instance_id(
        question: str,
        evidence: List[Dict[str, Any]],
    ) -> str:
        """
        Extract an EC2-style instance ID.

        First use previously collected evidence.
        Then fall back to the question.
        """

        for item in evidence:

            result = item.get(
                "result",
                {},
            )

            if isinstance(
                result,
                dict,
            ):

                instance_id = result.get(
                    "instance_id"
                )

                if instance_id:
                    return str(
                        instance_id
                    )

        match = re.search(
            r"\bi-[a-zA-Z0-9]+\b",
            question,
        )

        if match:
            return match.group(0)

        return ""

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def run(
        self,
        question: str,
        *,
        capability: str = "",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> CloudInvestigationGraphState:
        """
        Execute a complete cloud operations investigation.

        Production boundary validation is performed before
        LangGraph execution begins.

        Args:
            question:
                User's investigation question.

            capability:
                Optional capability hint.

            max_iterations:
                Maximum number of investigation iterations.
                Must be between 1 and 10.

        Returns:
            Final LangGraph investigation state.

        Raises:
            ValueError:
                If the question is empty or max_iterations
                is outside the supported range.
        """

        # ----------------------------------------------------------
        # Production boundary validation
        # ----------------------------------------------------------

        if not isinstance(
            question,
            str,
        ):

            raise ValueError(
                "Investigation question "
                "must be a string."
            )

        question = question.strip()

        if not question:

            raise ValueError(
                "Investigation question "
                "cannot be empty."
            )

        if not isinstance(
            max_iterations,
            int,
        ):

            raise ValueError(
                "max_iterations "
                "must be an integer."
            )

        if not (
            self.MIN_MAX_ITERATIONS
            <= max_iterations
            <= self.MAX_MAX_ITERATIONS
        ):

            raise ValueError(
                "max_iterations must be "
                "between 1 and 10."
            )

        # ----------------------------------------------------------
        # Initial state
        # ----------------------------------------------------------

        initial_state: CloudInvestigationGraphState = {
            "question": question,
            "capability": capability,
            "max_iterations": max_iterations,
        }

        # ----------------------------------------------------------
        # Start observability only after validation
        # ----------------------------------------------------------

        self.tracer.start_run(
            question=question,
            capability=capability,
        )

        try:

            result = self.graph.invoke(
                initial_state
            )

            self.tracer.end_run(
                result.get(
                    "status",
                    "completed",
                )
            )

        except Exception as exc:

            self.tracer.error(
                "langgraph_run",
                exc,
            )

            self.tracer.end_run(
                "failed"
            )

            raise

        # ----------------------------------------------------------
        # Attach observability information
        # ----------------------------------------------------------

        result[
            "trace_summary"
        ] = self.tracer.summary()

        result[
            "trace_events"
        ] = self.tracer.get_events()

        print(
            "\n[Observability Summary]"
        )

        print(
            f"Run ID: "
            f"{result['trace_summary']['run_id']}"
        )

        print(
            f"Events: "
            f"{result['trace_summary']['event_count']}"
        )

        print(
            f"Nodes: "
            f"{result['trace_summary']['node_count']}"
        )

        print(
            f"Tools: "
            f"{result['trace_summary']['tool_count']}"
        )

        print(
            f"Errors: "
            f"{result['trace_summary']['error_count']}"
        )

        print(
            "Node Duration (ms): "
            f"{result['trace_summary']['total_node_duration_ms']}"
        )

        return result


def create_cloud_operations_graph(
    llm: CohereClient | None = None,
):
    """
    Factory for creating the production
    LangGraph orchestration workflow.
    """

    return CloudOperationsLangGraph(
        llm=llm,
    )