from src.knowledge.knowledge_retriever import (
    KnowledgeRetriever,
)


def test_retriever_finds_ec2_runbook():

    retriever = KnowledgeRetriever()

    results = retriever.retrieve(
        "EC2 high CPU application unhealthy"
    )

    assert results

    assert (
        results[0]["source"]
        == "ec2_application_degradation.md"
    )


def test_retriever_returns_relevant_content():

    retriever = KnowledgeRetriever()

    results = retriever.retrieve(
        "database connection timeout"
    )

    assert results

    assert (
        "database connection"
        in results[0]["content"].lower()
    )


def test_retriever_returns_empty_for_empty_query():

    retriever = KnowledgeRetriever()

    assert (
        retriever.retrieve("")
        == []
    )


def test_retriever_respects_top_k():

    retriever = KnowledgeRetriever()

    results = retriever.retrieve(
        "EC2 application",
        top_k=1,
    )

    assert len(results) <= 1


def test_evidence_aware_retrieval():

    retriever = KnowledgeRetriever()

    findings = [
        "High CPU utilization detected: 92.4%.",
        "High memory utilization detected: 81.7%.",
        "Application status is unhealthy.",
    ]

    evidence = [
        {
            "tool": "get_application_logs",
            "result": {
                "logs": [
                    {
                        "level": "ERROR",
                        "message": (
                            "Database connection timeout "
                            "while processing request."
                        ),
                    },
                    {
                        "level": "WARN",
                        "message": (
                            "Connection pool utilization "
                            "reached 95%."
                        ),
                    },
                ]
            },
        }
    ]

    results = (
        retriever.retrieve_for_investigation(
            query=(
                "Why is instance unhealthy?"
            ),
            findings=findings,
            evidence=evidence,
        )
    )

    assert results

    result = results[0]

    assert (
        result["score"] > 0
    )

    assert (
        "matched_evidence_terms"
        in result
    )

    assert (
        "database"
        in result["matched_evidence_terms"]
    )


def test_evidence_can_increase_relevance():

    retriever = KnowledgeRetriever()

    question_results = (
        retriever.retrieve(
            "Why is instance unhealthy?"
        )
    )

    evidence_results = (
        retriever.retrieve_for_investigation(
            query=(
                "Why is instance unhealthy?"
            ),
            findings=[
                "High CPU utilization detected: 92.4%."
            ],
            evidence=[
                {
                    "tool": "get_application_logs",
                    "result": {
                        "message": (
                            "Database connection "
                            "timeout detected."
                        )
                    },
                }
            ],
        )
    )

    assert evidence_results

    assert (
        evidence_results[0]["score"]
        >= question_results[0]["score"]
    )