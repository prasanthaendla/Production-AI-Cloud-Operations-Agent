"""
Investigation Analyzer

Analyzes collected Cloud Operations evidence and
converts raw tool results into structured findings.

The analyzer does not call the LLM.

Its responsibility is to interpret known tool evidence
using deterministic investigation rules.
"""


class InvestigationAnalyzer:
    """
    Analyze investigation evidence and generate findings.

    The analyzer currently focuses on deterministic
    infrastructure signals such as:

    - High CPU utilization
    - High memory utilization
    - Unhealthy application status
    - Degraded instance health
    - Network problems
    - Recent deployment information

    Later this component can evolve to support
    LLM-assisted reasoning and RAG-based runbooks.
    """

    # --------------------------------------------------
    # Thresholds
    # --------------------------------------------------

    HIGH_CPU_THRESHOLD = 80.0

    HIGH_MEMORY_THRESHOLD = 80.0

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def analyze(
        self,
        evidence: list,
    ) -> list:
        """
        Analyze collected evidence.

        Args:
            evidence:
                Evidence collected by InvestigationState.

        Returns:
            List of structured investigation findings.
        """

        findings = []

        for evidence_item in evidence:

            tool_name = evidence_item.get(
                "tool"
            )

            result = evidence_item.get(
                "result"
            )

            if not isinstance(result, dict):
                continue

            if tool_name == "get_instance_health":

                findings.extend(
                    self._analyze_instance_health(
                        result
                    )
                )

            elif tool_name == "get_application_logs":

                findings.extend(
                    self._analyze_application_logs(
                        result
                    )
                )

            elif tool_name == "get_recent_deployments":

                findings.extend(
                    self._analyze_deployments(
                        result
                    )
                )

        return findings

    # --------------------------------------------------
    # Instance Health Analysis
    # --------------------------------------------------

    def _analyze_instance_health(
        self,
        result: dict,
    ) -> list:
        """
        Analyze instance health evidence.
        """

        findings = []

        cpu = result.get(
            "cpu_utilization"
        )

        memory = result.get(
            "memory_utilization"
        )

        health = result.get(
            "health"
        )

        application_status = result.get(
            "application_status"
        )

        network_status = result.get(
            "network_status"
        )

        # --------------------------------------------------
        # CPU
        # --------------------------------------------------

        if (
            isinstance(cpu, (int, float))
            and cpu >= self.HIGH_CPU_THRESHOLD
        ):
            findings.append(
                (
                    f"High CPU utilization detected: "
                    f"{cpu}%."
                )
            )

        # --------------------------------------------------
        # Memory
        # --------------------------------------------------

        if (
            isinstance(memory, (int, float))
            and memory >= self.HIGH_MEMORY_THRESHOLD
        ):
            findings.append(
                (
                    f"High memory utilization detected: "
                    f"{memory}%."
                )
            )

        # --------------------------------------------------
        # Instance health
        # --------------------------------------------------

        if health in {
            "degraded",
            "unhealthy",
            "failed",
        }:
            findings.append(
                (
                    f"Instance health is "
                    f"{health}."
                )
            )

        # --------------------------------------------------
        # Application health
        # --------------------------------------------------

        if application_status in {
            "unhealthy",
            "degraded",
            "failed",
        }:
            findings.append(
                (
                    f"Application status is "
                    f"{application_status}."
                )
            )

        # --------------------------------------------------
        # Network
        # --------------------------------------------------

        if network_status not in {
            None,
            "normal",
            "healthy",
            "available",
        }:
            findings.append(
                (
                    f"Network status is "
                    f"{network_status}."
                )
            )

        return findings

    # --------------------------------------------------
    # Application Log Analysis
    # --------------------------------------------------

    def _analyze_application_logs(
        self,
        result: dict,
    ) -> list:
        """
        Analyze application log evidence.

        The current demo tools may return different
        structures, so the analyzer handles common
        log-related fields conservatively.
        """

        findings = []

        errors = result.get(
            "errors"
        )

        warnings = result.get(
            "warnings"
        )

        if errors:

            findings.append(
                (
                    f"Application errors detected: "
                    f"{errors}"
                )
            )

        if warnings:

            findings.append(
                (
                    f"Application warnings detected: "
                    f"{warnings}"
                )
            )

        return findings

    # --------------------------------------------------
    # Deployment Analysis
    # --------------------------------------------------

    def _analyze_deployments(
        self,
        result: dict,
    ) -> list:
        """
        Analyze recent deployment evidence.
        """

        findings = []

        deployment_status = result.get(
            "status"
        )

        if deployment_status in {
            "failed",
            "failure",
        }:
            findings.append(
                (
                    "A recent deployment failed "
                    "and may be related to the incident."
                )
            )

        return findings