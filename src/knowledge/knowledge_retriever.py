"""
Operational Knowledge Retriever

Stage 14.4:
Evidence-aware retrieval from operational knowledge.

The retriever can use:
- Original user question
- Investigation findings
- Investigation evidence

The retrieval interface remains intentionally simple so the
underlying implementation can later be upgraded to semantic
vector retrieval without changing the agent contract.
"""

from pathlib import Path
import re


class KnowledgeRetriever:
    """
    Retrieve relevant operational knowledge from runbooks.
    """

    def __init__(self, knowledge_dir=None):

        if knowledge_dir is None:
            knowledge_dir = (
                Path(__file__).resolve().parent
                / "runbooks"
            )

        self.knowledge_dir = Path(
            knowledge_dir
        )

    # ==================================================
    # Public API
    # ==================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list:
        """
        Retrieve knowledge using a query.
        """

        return self.retrieve_for_investigation(
            query=query,
            findings=[],
            evidence=[],
            top_k=top_k,
        )

    def retrieve_for_investigation(
        self,
        query: str,
        findings: list = None,
        evidence: list = None,
        top_k: int = 3,
    ) -> list:
        """
        Retrieve knowledge using the current investigation
        context.

        Relevance is calculated from:

        1. Original question
        2. Investigation findings
        3. Tool evidence

        Evidence and findings therefore influence retrieval
        after the investigation discovers additional facts.
        """

        if (
            not isinstance(query, str)
            or not query.strip()
        ):
            return []

        if top_k <= 0:
            return []

        findings = findings or []
        evidence = evidence or []

        documents = self._load_documents()

        if not documents:
            return []

        query_text = query

        findings_text = self._findings_to_text(
            findings
        )

        evidence_text = self._evidence_to_text(
            evidence
        )

        query_terms = self._tokenize(
            query_text
        )

        finding_terms = self._tokenize(
            findings_text
        )

        evidence_terms = self._tokenize(
            evidence_text
        )

        if not (
            query_terms
            or finding_terms
            or evidence_terms
        ):
            return []

        results = []

        for source, content in documents:

            document_terms = self._tokenize(
                content
            )

            query_matches = (
                query_terms
                & document_terms
            )

            finding_matches = (
                finding_terms
                & document_terms
            )

            evidence_matches = (
                evidence_terms
                & document_terms
            )

            # Original question has the lowest weight.
            query_score = len(
                query_matches
            )

            # Findings represent discovered facts.
            finding_score = (
                len(finding_matches) * 2
            )

            # Actual tool evidence is the strongest signal.
            evidence_score = (
                len(evidence_matches) * 3
            )

            score = (
                query_score
                + finding_score
                + evidence_score
            )

            if score <= 0:
                continue

            results.append(
                {
                    "source": source,
                    "content": content,
                    "score": score,
                    "matched_query_terms": sorted(
                        query_matches
                    ),
                    "matched_finding_terms": sorted(
                        finding_matches
                    ),
                    "matched_evidence_terms": sorted(
                        evidence_matches
                    ),
                }
            )

        results.sort(
            key=lambda item: (
                -item["score"],
                item["source"],
            )
        )

        return results[:top_k]

    # ==================================================
    # Document Loading
    # ==================================================

    def _load_documents(self) -> list:
        """
        Load markdown documents from the runbook directory.
        """

        if not self.knowledge_dir.exists():
            return []

        documents = []

        for path in sorted(
            self.knowledge_dir.glob("*.md")
        ):

            try:
                content = path.read_text(
                    encoding="utf-8"
                )

            except OSError:
                continue

            if content.strip():

                documents.append(
                    (
                        path.name,
                        content,
                    )
                )

        return documents

    # ==================================================
    # Context Extraction
    # ==================================================

    @staticmethod
    def _findings_to_text(
        findings: list,
    ) -> str:
        """
        Convert investigation findings into searchable text.
        """

        if not findings:
            return ""

        return "\n".join(
            str(finding)
            for finding in findings
            if isinstance(
                finding,
                str,
            )
        )

    @staticmethod
    def _evidence_to_text(
        evidence: list,
    ) -> str:
        """
        Convert tool evidence into searchable text.

        Evidence can contain nested dictionaries/lists.
        Converting it to text keeps the retriever independent
        of individual cloud-tool schemas.
        """

        if not evidence:
            return ""

        evidence_text = []

        for item in evidence:

            if isinstance(
                item,
                dict,
            ):

                evidence_text.append(
                    str(item)
                )

            else:

                evidence_text.append(
                    str(item)
                )

        return "\n".join(
            evidence_text
        )

    # ==================================================
    # Tokenization
    # ==================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> set:
        """
        Normalize text into searchable terms.
        """

        if not isinstance(
            text,
            str,
        ):
            return set()

        tokens = re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )

        return {
            token
            for token in tokens
            if len(token) > 1
        }