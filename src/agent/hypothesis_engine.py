"""
Investigation Hypothesis Engine

Converts deterministic investigation findings into
structured hypotheses that can guide additional
investigation.

The hypothesis engine does not call the LLM.

Its responsibility is to identify possible explanations
from known evidence without claiming that a hypothesis
is a confirmed root cause.
"""


class HypothesisEngine:
    """
    Generate investigation hypotheses from findings.

    A hypothesis represents a possible explanation for
    an incident.

    It is intentionally different from a finding:

        Finding:
            CPU utilization is 92.4%.

        Hypothesis:
            Resource saturation may be contributing
            to application degradation.

    Hypotheses are not treated as confirmed root causes.
    Additional evidence should be collected before
    making a final root-cause assessment.
    """

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def generate(
        self,
        findings: list,
    ) -> list:
        """
        Generate hypotheses from investigation findings.

        Args:
            findings:
                Findings produced by InvestigationAnalyzer.

        Returns:
            List of structured hypothesis dictionaries.
        """

        hypotheses = []

        # --------------------------------------------------
        # Resource saturation hypothesis
        # --------------------------------------------------

        has_high_cpu = self._contains_finding(
            findings,
            "High CPU utilization detected:",
        )

        has_high_memory = self._contains_finding(
            findings,
            "High memory utilization detected:",
        )

        application_unhealthy = self._contains_finding(
            findings,
            "Application status is unhealthy.",
        )

        if (
            (has_high_cpu or has_high_memory)
            and application_unhealthy
        ):

            hypotheses.append(
                {
                    "hypothesis": (
                        "Resource saturation may be "
                        "contributing to application "
                        "degradation."
                    ),
                    "confidence": "medium",
                    "supporting_findings": [
                        finding
                        for finding in findings
                        if (
                            "High CPU utilization detected:"
                            in finding
                            or
                            "High memory utilization detected:"
                            in finding
                            or
                            "Application status is unhealthy."
                            in finding
                        )
                    ],
                    "requires_validation": True,
                }
            )

        # --------------------------------------------------
        # Instance degradation hypothesis
        # --------------------------------------------------

        instance_degraded = self._contains_finding(
            findings,
            "Instance health is degraded.",
        )

        if instance_degraded:

            hypotheses.append(
                {
                    "hypothesis": (
                        "The degraded instance health "
                        "may be contributing to the "
                        "observed incident."
                    ),
                    "confidence": "medium",
                    "supporting_findings": [
                        finding
                        for finding in findings
                        if (
                            "Instance health is degraded."
                            in finding
                        )
                    ],
                    "requires_validation": True,
                }
            )

        # --------------------------------------------------
        # Network hypothesis
        # --------------------------------------------------

        network_problem = (
            self._contains_finding(
                findings,
                "Network status is",
            )
        )

        if network_problem:

            hypotheses.append(
                {
                    "hypothesis": (
                        "A network-related problem may "
                        "be contributing to the incident."
                    ),
                    "confidence": "medium",
                    "supporting_findings": [
                        finding
                        for finding in findings
                        if (
                            "Network status is"
                            in finding
                        )
                    ],
                    "requires_validation": True,
                }
            )

        # --------------------------------------------------
        # Application log hypothesis
        # --------------------------------------------------

        application_errors = (
            self._contains_finding(
                findings,
                "Application errors detected:",
            )
        )

        if application_errors:

            hypotheses.append(
                {
                    "hypothesis": (
                        "Application errors may be "
                        "contributing to the observed "
                        "service degradation."
                    ),
                    "confidence": "medium",
                    "supporting_findings": [
                        finding
                        for finding in findings
                        if (
                            "Application errors detected:"
                            in finding
                        )
                    ],
                    "requires_validation": True,
                }
            )

        # --------------------------------------------------
        # Deployment hypothesis
        # --------------------------------------------------

        deployment_failure = (
            self._contains_finding(
                findings,
                "A recent deployment failed",
            )
        )

        if deployment_failure:

            hypotheses.append(
                {
                    "hypothesis": (
                        "A recent failed deployment may "
                        "be related to the incident."
                    ),
                    "confidence": "medium",
                    "supporting_findings": [
                        finding
                        for finding in findings
                        if (
                            "A recent deployment failed"
                            in finding
                        )
                    ],
                    "requires_validation": True,
                }
            )

        return hypotheses

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------

    @staticmethod
    def _contains_finding(
        findings: list,
        text: str,
    ) -> bool:
        """
        Check whether any finding contains the
        specified text.
        """

        return any(
            text in finding
            for finding in findings
            if isinstance(
                finding,
                str,
            )
        )