"""
Confidence Engine

Evaluates confidence in investigation root-cause assessments
based on supporting and contradicting evidence.

The engine does not determine the root cause itself.
It evaluates the strength of an existing assessment and
explicitly communicates uncertainty.
"""


class ConfidenceEngine:
    """
    Evaluate confidence for root-cause assessments.

    Confidence levels:

        HIGH
        MEDIUM
        LOW

    The engine intentionally remains deterministic so that
    confidence decisions are explainable and testable.
    """

    HIGH_THRESHOLD = 3
    MEDIUM_THRESHOLD = 1

    def evaluate(
        self,
        assessment: dict,
    ) -> dict:
        """
        Evaluate confidence for one root-cause assessment.

        Expected assessment structure:

        {
            "root_cause": "...",
            "score": 4,
            "supporting_evidence": [...],
            "contradicting_evidence": [...]
        }
        """

        if not isinstance(
            assessment,
            dict,
        ):
            return self._uncertain_result(
                "The root-cause assessment is unavailable."
            )

        root_cause = assessment.get(
            "root_cause"
        )

        supporting = assessment.get(
            "supporting_evidence",
            [],
        )

        contradicting = assessment.get(
            "contradicting_evidence",
            [],
        )

        if not isinstance(
            supporting,
            list,
        ):
            supporting = []

        if not isinstance(
            contradicting,
            list,
        ):
            contradicting = []

        support_count = len(
            supporting
        )

        contradiction_count = len(
            contradicting
        )

        score = assessment.get(
            "score",
            0,
        )

        if not isinstance(
            score,
            (int, float),
        ):
            score = 0

        # --------------------------------------------------
        # Strongly confirmed
        # --------------------------------------------------

        if (
            score >= self.HIGH_THRESHOLD
            and support_count >= 2
            and contradiction_count == 0
        ):

            confidence_level = "HIGH"

            confidence_score = 0.90

            uncertainty = (
                "The available evidence strongly "
                "supports this root-cause assessment."
            )

        # --------------------------------------------------
        # Moderate confidence
        # --------------------------------------------------

        elif (
            support_count >= 1
            and contradiction_count <= 1
        ):

            confidence_level = "MEDIUM"

            confidence_score = 0.65

            uncertainty = (
                "The available evidence supports "
                "this root-cause assessment, but "
                "additional evidence may be required "
                "for confirmation."
            )

        # --------------------------------------------------
        # Low confidence
        # --------------------------------------------------

        else:

            confidence_level = "LOW"

            confidence_score = 0.30

            uncertainty = (
                "The available evidence is insufficient "
                "to confidently confirm the root cause. "
                "Further investigation is recommended."
            )

        return {
            "root_cause": root_cause,
            "confidence_level": confidence_level,
            "confidence_score": confidence_score,
            "supporting_evidence_count": (
                support_count
            ),
            "contradicting_evidence_count": (
                contradiction_count
            ),
            "uncertainty": uncertainty,
        }

    # ======================================================
    # Multiple assessments
    # ======================================================

    def evaluate_all(
        self,
        assessments: list,
    ) -> list:
        """
        Evaluate multiple root-cause assessments.
        """

        if not isinstance(
            assessments,
            list,
        ):
            return []

        return [
            self.evaluate(
                assessment
            )
            for assessment in assessments
            if isinstance(
                assessment,
                dict,
            )
        ]

    # ======================================================
    # Uncertain result
    # ======================================================

    @staticmethod
    def _uncertain_result(
        message: str,
    ) -> dict:
        """
        Return an explicit low-confidence result.
        """

        return {
            "root_cause": None,
            "confidence_level": "LOW",
            "confidence_score": 0.0,
            "supporting_evidence_count": 0,
            "contradicting_evidence_count": 0,
            "uncertainty": message
            + " Further investigation is recommended.",
        }