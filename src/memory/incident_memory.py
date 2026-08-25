"""
Historical Incident Memory

Stage 15:
Persistent memory for completed cloud investigations.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import re


class IncidentMemory:
    """Persistent historical incident memory."""

    def __init__(self, storage_path=None):
        if storage_path is None:
            storage_path = (
                Path(__file__).resolve().parent
                / "incidents.json"
            )

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.storage_path.exists():
            self._write([])

    def save_incident(
        self,
        question: str,
        capability: str,
        findings: list,
        hypotheses: list,
        root_cause_assessment: dict,
        confidence_assessment: dict,
    ) -> dict:
        """Persist a compact completed investigation summary."""

        incidents = self._read()

        incident = {
            "incident_id": self._next_incident_id(
                incidents
            ),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "question": question,
            "capability": capability,
            "findings": [
                finding
                for finding in findings
                if isinstance(finding, str)
            ],
            "hypotheses": [
                hypothesis
                for hypothesis in hypotheses
                if isinstance(hypothesis, dict)
            ],
            "root_cause": (
                root_cause_assessment.get("root_cause")
                if isinstance(
                    root_cause_assessment,
                    dict,
                )
                else None
            ),
            "root_cause_score": (
                root_cause_assessment.get("score", 0)
                if isinstance(
                    root_cause_assessment,
                    dict,
                )
                else 0
            ),
            "confidence_level": (
                confidence_assessment.get(
                    "confidence_level"
                )
                if isinstance(
                    confidence_assessment,
                    dict,
                )
                else None
            ),
            "confidence_score": (
                confidence_assessment.get(
                    "confidence_score"
                )
                if isinstance(
                    confidence_assessment,
                    dict,
                )
                else None
            ),
        }

        incidents.append(incident)
        self._write(incidents)

        return incident

    def retrieve_similar(
        self,
        question: str,
        findings: list = None,
        capability: str = None,
        top_k: int = 3,
    ) -> list:
        """Retrieve historically similar incidents."""

        if (
            not isinstance(question, str)
            or not question.strip()
            or top_k <= 0
        ):
            return []

        findings = findings or []

        query_text = " ".join(
            [
                question,
                " ".join(
                    str(finding)
                    for finding in findings
                    if isinstance(finding, str)
                ),
            ]
        )

        query_terms = self._tokenize(query_text)

        if not query_terms:
            return []

        results = []

        for incident in self._read():
            if not isinstance(incident, dict):
                continue

            incident_text = " ".join(
                [
                    str(incident.get("question", "")),
                    str(incident.get("capability", "")),
                    " ".join(
                        incident.get("findings", [])
                    ),
                    str(
                        incident.get(
                            "root_cause",
                            "",
                        )
                    ),
                ]
            )

            incident_terms = self._tokenize(
                incident_text
            )

            overlap = (
                query_terms & incident_terms
            )

            score = len(overlap)

            if (
                capability
                and incident.get("capability")
                == capability
            ):
                score += 2

            if score <= 0:
                continue

            result = dict(incident)
            result["similarity_score"] = score
            result["matched_terms"] = sorted(
                overlap
            )

            results.append(result)

        results.sort(
            key=lambda item: (
                -item["similarity_score"],
                item.get("timestamp", ""),
            )
        )

        return results[:top_k]

    def count(self) -> int:
        """Return number of stored incidents."""
        return len(self._read())

    def _read(self) -> list:
        try:
            data = json.loads(
                self.storage_path.read_text(
                    encoding="utf-8"
                )
            )
            return (
                data
                if isinstance(data, list)
                else []
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return []

    def _write(self, incidents: list):
        self.storage_path.write_text(
            json.dumps(
                incidents,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _next_incident_id(
        incidents: list,
    ) -> str:
        return (
            f"INC-{len(incidents) + 1:05d}"
        )

    @staticmethod
    def _tokenize(text: str) -> set:
        tokens = re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )

        stop_words = {
            "the",
            "is",
            "are",
            "was",
            "were",
            "why",
            "what",
            "how",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "instance",
            "application",
        }

        return {
            token
            for token in tokens
            if len(token) > 2
            and token not in stop_words
        }

