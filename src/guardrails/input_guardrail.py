"""
Input Guardrail

Controls whether a user request is within the scope
of the AI Cloud Operations Agent.

The guardrail runs before the LLM is called.
"""

import re


class InputGuardrail:
    """
    Validate whether a user question is relevant
    to cloud operations.
    """

    CLOUD_OPERATIONS_TERMS = {
        "cloud",
        "aws",
        "ec2",
        "instance",
        "server",
        "infrastructure",
        "cpu",
        "memory",
        "utilization",
        "health",
        "healthy",
        "unhealthy",
        "degraded",
        "application",
        "logs",
        "log",
        "error",
        "errors",
        "warning",
        "warnings",
        "http",
        "500",
        "deployment",
        "deploy",
        "release",
        "incident",
        "outage",
        "monitoring",
        "metric",
        "metrics",
        "performance",
        "latency",
        "timeout",
        "database",
        "connection",
        "network",
        "service",
        "availability",
        "troubleshoot",
        "troubleshooting",
        "root",
        "cause",
        "operations",
        "sre",
        "devops",
    }

    CLOUD_CONTEXT_PATTERNS = [
        r"\bi-[a-z0-9-]+\b",
        r"\binstance\s+\w+",
        r"\bec2\b",
        r"\bcloudwatch\b",
        r"\baws\b",
        r"\bhttp\s*[45]\d\d\b",
        r"\bcpu\s*(usage|utilization)?\b",
        r"\bmemory\s*(usage|utilization)?\b",
    ]

    def check(self, question: str) -> bool:
        """
        Determine whether a question is within the
        cloud operations domain.

        Args:
            question:
                User's question.

        Returns:
            True if the question is considered
            cloud-operations related.
        """

        if not question or not question.strip():
            return False

        normalized_question = question.lower().strip()

        words = set(
            re.findall(
                r"\b[a-z0-9-]+\b",
                normalized_question,
            )
        )

        # Direct keyword match
        if words.intersection(
            self.CLOUD_OPERATIONS_TERMS
        ):
            return True

        # Context pattern match
        for pattern in self.CLOUD_CONTEXT_PATTERNS:
            if re.search(
                pattern,
                normalized_question,
            ):
                return True

        return False

    def get_rejection_message(self) -> str:
        """
        Return the standard response for an
        out-of-scope request.
        """

        return (
            "I'm an AI Cloud Operations Agent focused on "
            "cloud infrastructure health, incidents, "
            "application logs, deployments, monitoring, "
            "performance, and troubleshooting. "
            "Please ask a question related to cloud "
            "operations."
        )