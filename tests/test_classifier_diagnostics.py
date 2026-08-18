from src.guardrails.semantic_classifier import (
    SemanticScopeClassifier,
)


def test_classifier_diagnostics():

    classifier = SemanticScopeClassifier()

    questions = [
        "What is a route table?",
        "What is machine learning?",
        "How do I learn Java?",
        "What are Oracle 21c new features?",
        "What is a VPC?",
        "Why is my EC2 CPU utilization high?",
        "What is Agentic AI?",
        "Why is my application unhealthy?",
        "Why did my cloud deployment fail?",
    ]

    for question in questions:

        result = classifier.classify(question)

        print("\n" + "=" * 75)
        print(
            f"QUESTION: {question}"
        )
        print("=" * 75)

        print(
            f"Decision: "
            f"{result['category']}"
        )

        print(
            f"Positive similarity: "
            f"{result['positive_similarity']}"
        )

        print(
            f"Negative similarity: "
            f"{result['negative_similarity']}"
        )

        print(
            f"Margin: "
            f"{result['margin']}"
        )

        print(
            f"Matched positive domain: "
            f"{result['matched_domain']}"
        )

        print("\nPositive similarities:")

        for item in result[
            "positive_similarities"
        ]:

            print(
                f"  {item['domain']:<30} "
                f"{item['similarity']:.4f}"
            )

        print("\nNegative similarities:")

        for item in result[
            "negative_similarities"
        ]:

            print(
                f"  {item['domain']:<35} "
                f"{item['similarity']:.4f}"
            )