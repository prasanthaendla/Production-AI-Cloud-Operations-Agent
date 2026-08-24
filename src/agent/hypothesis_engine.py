"""
Investigation Hypothesis Engine

Converts deterministic investigation findings into
structured hypotheses and identifies additional
evidence that may be required to validate them.
"""


class HypothesisEngine:
    """
    Generate investigation hypotheses from findings.

    A hypothesis represents a possible explanation for
    an incident.

    The engine also identifies recommended evidence
    that can be collected to validate each hypothesis.

    It does not claim that a hypothesis is a confirmed
    root cause.
    """

    HIGH_CPU_TEXT = (
        "High CPU utilization detected:"
    )

    HIGH_MEMORY_TEXT = (
        "High memory utilization detected:"
    )

    APPLICATION_UNHEALTHY_TEXT = (
        "Application status is unhealthy."
    )

    INSTANCE_DEGRADED_TEXT = (
        "Instance health is degraded."
    )

    NETWORK_PROBLEM_TEXT = (
        "Network status is"
    )

    APPLICATION_ERROR_TEXT = (
        "Application errors detected:"
    )

    DEPLOYMENT_FAILURE_TEXT = (
        "A recent deployment failed"
    )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def generate(
        self,
        findings: list,
    ) -> list:
        """
        Generate hypotheses from investigation findings.
        """

        hypotheses = []

        # --------------------------------------------------
        # Resource saturation
        # --------------------------------------------------

        has_high_cpu = self._contains_finding(
            findings,
            self.HIGH_CPU_TEXT,
        )

        has_high_memory = self._contains_finding(
            findings,
            self.HIGH_MEMORY_TEXT,
        )

        application_unhealthy = self._contains_finding(
            findings,
            self.APPLICATION_UNHEALTHY_TEXT,
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
                            self.HIGH_CPU_TEXT
                            in finding
                            or
                            self.HIGH_MEMORY_TEXT
                            in finding
                            or
                            self.APPLICATION_UNHEALTHY_TEXT
                            in finding
                        )
                    ],
                    "requires_validation": True,
                    "recommended_evidence": [
                        "application_logs",
                        "deployments",
                    ],
                }
            )

        # --------------------------------------------------
        # Instance degradation
        # --------------------------------------------------

        instance_degraded = self._contains_finding(
            findings,
            self.INSTANCE_DEGRADED_TEXT,
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
                            self.INSTANCE_DEGRADED_TEXT
                            in finding
                        )
                    ],
                    "requires_validation": True,
                    "recommended_evidence": [
                        "application_logs",
                        "deployments",
                    ],
                }
            )

        # --------------------------------------------------
        # Network problem
        # --------------------------------------------------

        network_problem = self._contains_finding(
            findings,
            self.NETWORK_PROBLEM_TEXT,
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
                            self.NETWORK_PROBLEM_TEXT
                            in finding
                        )
                    ],
                    "requires_validation": True,
                    "recommended_evidence": [
                        "application_logs",
                    ],
                }
            )

        # --------------------------------------------------
        # Application errors
        # --------------------------------------------------

        application_errors = self._contains_finding(
            findings,
            self.APPLICATION_ERROR_TEXT,
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
                            self.APPLICATION_ERROR_TEXT
                            in finding
                        )
                    ],
                    "requires_validation": True,
                    "recommended_evidence": [
                        "deployments",
                    ],
                }
            )

        # --------------------------------------------------
        # Deployment failure
        # --------------------------------------------------

        deployment_failure = self._contains_finding(
            findings,
            self.DEPLOYMENT_FAILURE_TEXT,
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
                            self.DEPLOYMENT_FAILURE_TEXT
                            in finding
                        )
                    ],
                    "requires_validation": True,
                    "recommended_evidence": [
                        "application_logs",
                    ],
                }
            )

        return hypotheses

    # --------------------------------------------------
    # Evidence Recommendations
    # --------------------------------------------------

    def get_recommended_evidence(
        self,
        hypotheses: list,
    ) -> list:
        """
        Return unique evidence types recommended for
        validating the generated hypotheses.
        """

        evidence_types = []

        for hypothesis in hypotheses:

            if not isinstance(
                hypothesis,
                dict,
            ):
                continue

            if not hypothesis.get(
                "requires_validation",
                False,
            ):
                continue

            for evidence_type in hypothesis.get(
                "recommended_evidence",
                [],
            ):

                if (
                    evidence_type
                    not in evidence_types
                ):

                    evidence_types.append(
                        evidence_type
                    )

        return evidence_types

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------

    @staticmethod
    def _contains_finding(
        findings: list,
        text: str,
    ) -> bool:
        """
        Check whether any finding contains
        the specified text.
        """

        return any(
            text in finding
            for finding in findings
            if isinstance(
                finding,
                str,
            )
        )