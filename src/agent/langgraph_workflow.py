"""
LangGraph orchestration for the AI Cloud Operations Agent.

Stage 16 - LangGraph Orchestration

Responsibilities:
- Provide explicit stateful workflow orchestration.
- Preserve the existing CloudOperationsAgent investigation logic.
- Use the existing agent as the investigation engine.
- Keep orchestration separate from investigation intelligence.
- Provide controlled workflow states and safety limits.

The existing CloudOperationsAgent remains responsible for:
- Semantic guardrails
- Capability routing
- Investigation state
- Tool calling
- Evidence collection
- Investigation analysis
- Hypothesis generation
- Evidence-driven investigation
- Root cause assessment
- Confidence assessment
- Operational knowledge / RAG
- Historical incident memory
- Final answer generation
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.agent import CloudOperationsAgent


class CloudInvestigationGraphState(TypedDict, total=False):
    """
    State maintained by the LangGraph orchestration layer.
    """

    question: str

    capability: str

    max_iterations: int

    iteration: int

    status: str

    error: str

    final_answer: str

    tool_calls: List[Dict[str, Any]]

    evidence: List[Dict[str, Any]]

    findings: List[str]

    hypotheses: List[Any]

    root_cause: str

    root_cause_score: int

    confidence_level: str

    confidence_score: float

    knowledge: List[Dict[str, Any]]

    historical_incidents: List[Dict[str, Any]]


class CloudOperationsLangGraph:
    """
    LangGraph orchestration layer.

    IMPORTANT:

    This class does not duplicate the investigation intelligence
    already implemented inside CloudOperationsAgent.

    Instead, LangGraph controls the lifecycle of the investigation
    while CloudOperationsAgent remains responsible for the actual
    investigation.

    This separation allows us to later replace the mock tool layer
    with real AWS adapters without redesigning the orchestration.
    """

    DEFAULT_MAX_ITERATIONS = 5

    def __init__(
        self,
        agent: CloudOperationsAgent | None = None,
    ) -> None:

        self.agent = agent or CloudOperationsAgent()

        self.graph = self._build_graph()

    # ==============================================================
    # GRAPH CONSTRUCTION
    # ==============================================================

    def _build_graph(self):
        """
        Build the LangGraph workflow.

        Workflow:

            START
              ↓
          initialize
              ↓
          investigate
              ↓
            complete
              ↓
             END
        """

        workflow = StateGraph(
            CloudInvestigationGraphState
        )

        workflow.add_node(
            "initialize",
            self._initialize,
        )

        workflow.add_node(
            "investigate",
            self._investigate,
        )

        workflow.add_node(
            "complete",
            self._complete,
        )

        workflow.add_edge(
            START,
            "initialize",
        )

        workflow.add_edge(
            "initialize",
            "investigate",
        )

        workflow.add_edge(
            "investigate",
            "complete",
        )

        workflow.add_edge(
            "complete",
            END,
        )

        return workflow.compile()

    # ==============================================================
    # INITIALIZE
    # ==============================================================

    def _initialize(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:
        """
        Initialize the orchestration state.
        """

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
                "iteration": 0,
                "final_answer": "",
            }

        max_iterations = state.get(
            "max_iterations",
            self.DEFAULT_MAX_ITERATIONS,
        )

        return {
            "question": question,
            "capability": state.get(
                "capability",
                "",
            ),
            "max_iterations": max_iterations,
            "iteration": 0,
            "status": "initialized",
            "error": "",
            "final_answer": "",
            "tool_calls": [],
            "evidence": [],
            "findings": [],
            "hypotheses": [],
            "knowledge": [],
            "historical_incidents": [],
        }

    # ==============================================================
    # INVESTIGATION
    # ==============================================================

    def _investigate(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:
        """
        Execute the existing CloudOperationsAgent.

        The existing agent already contains the complete
        investigation workflow.

        LangGraph is responsible for orchestrating the lifecycle,
        while CloudOperationsAgent remains the investigation engine.
        """

        if state.get("status") == "failed":

            return {
                "status": "failed",
                "iteration": state.get(
                    "iteration",
                    0,
                ),
            }

        question = state.get(
            "question",
            "",
        )

        if not question:

            return {
                "status": "failed",
                "error": (
                    "Investigation question "
                    "is missing."
                ),
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
                    "Maximum LangGraph "
                    "iteration limit exceeded."
                ),
                "iteration": iteration,
            }

        try:

            final_answer = self.agent.run(
                question,
            )

            return {
                "status": "investigation_completed",
                "iteration": iteration,
                "final_answer": final_answer,
            }

        except Exception as exc:

            return {
                "status": "failed",
                "iteration": iteration,
                "error": str(exc),
            }

    # ==============================================================
    # COMPLETE
    # ==============================================================

    def _complete(
        self,
        state: CloudInvestigationGraphState,
    ) -> Dict[str, Any]:
        """
        Complete the LangGraph workflow.
        """

        if state.get("status") == "failed":

            return {
                "status": "failed",
                "final_answer": state.get(
                    "final_answer",
                    "",
                ),
                "error": state.get(
                    "error",
                    "Unknown investigation error.",
                ),
            }

        return {
            "status": "completed",
            "final_answer": state.get(
                "final_answer",
                "",
            ),
        }

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
        Execute the LangGraph workflow.

        Parameters
        ----------
        question:
            User's cloud operations question.

        capability:
            Optional capability supplied by an external caller.

        max_iterations:
            Maximum orchestration iterations.

        Returns
        -------
        CloudInvestigationGraphState
            Final LangGraph state.
        """

        initial_state: CloudInvestigationGraphState = {
            "question": question,
            "capability": capability,
            "max_iterations": max_iterations,
        }

        return self.graph.invoke(
            initial_state,
        )


def create_cloud_operations_graph(
    agent: CloudOperationsAgent | None = None,
):
    """
    Factory function for creating the LangGraph
    orchestration layer.
    """

    return CloudOperationsLangGraph(
        agent=agent,
    )