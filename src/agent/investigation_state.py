"""
Investigation State

Maintains the state of a Cloud Operations investigation
throughout the agent execution.
"""


class InvestigationState:
    """
    Represents the state of one cloud investigation.

    The state is intentionally simple for now.
    Later this can evolve into a structured state model
    when we introduce LangGraph.
    """

    def __init__(
        self,
        question: str,
        capability: str,
    ):
        """
        Initialize investigation state.
        """

        self.question = question

        self.capability = capability

        self.iterations = 0

        self.tool_calls = []

        self.evidence = []

        self.findings = []

    # --------------------------------------------------
    # Iteration tracking
    # --------------------------------------------------

    def record_iteration(
        self,
        iteration: int,
    ):
        """
        Record the current agent iteration.
        """

        self.iterations = iteration

    # --------------------------------------------------
    # Tool call tracking
    # --------------------------------------------------

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict,
    ):
        """
        Record a tool requested by the agent.
        """

        self.tool_calls.append(
            {
                "tool": tool_name,
                "arguments": arguments,
            }
        )

    # --------------------------------------------------
    # Evidence tracking
    # --------------------------------------------------

    def record_evidence(
        self,
        tool_name: str,
        arguments: dict,
        result,
    ):
        """
        Record evidence returned by a tool.
        """

        self.evidence.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

    # --------------------------------------------------
    # Findings
    # --------------------------------------------------

    def add_finding(
        self,
        finding: str,
    ):
        """
        Add an investigation finding.
        """

        self.findings.append(
            finding
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def summary(self) -> dict:
        """
        Return a structured investigation summary.
        """

        return {
            "question": self.question,
            "capability": self.capability,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "evidence": self.evidence,
            "findings": self.findings,
        }