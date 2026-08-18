"""
Semantic Scope Classifier

Uses Cohere embeddings to determine whether a user
question belongs to the supported Cloud Operations
domain.

The classifier uses positive and negative semantic
intent prototypes rather than maintaining a large
keyword or question list.

Positive prototypes represent common Cloud Operations
intents.

Negative prototypes represent domains that should remain
outside the Cloud Operations Agent.
"""

import math
import os

import cohere


class SemanticScopeClassifier:
    """
    Semantic classifier for the Cloud Operations Agent.

    The classifier compares a user's question against:

    1. Positive Cloud Operations intent prototypes.
    2. Negative out-of-scope prototypes.

    The final decision uses:

    - Positive semantic similarity.
    - Negative semantic similarity.
    - Semantic margin between positive and negative scores.
    """

    # ============================================================
    # POSITIVE CLOUD OPERATIONS INTENT PROTOTYPES
    # ============================================================

    CLOUD_DOMAIN_PROTOTYPES = [
        (
            "cloud infrastructure",
            (
                "Cloud infrastructure concepts including AWS, "
                "Amazon Web Services, GCP, Google Cloud, Azure, "
                "EC2, virtual machines, servers, compute resources, "
                "storage, availability, infrastructure health and "
                "cloud resource management."
            ),
        ),
        (
            "cloud instance health",
            (
                "Investigating the health and status of cloud "
                "instances, EC2 instances, virtual machines and "
                "production servers. Questions about unhealthy "
                "instances, degraded instances, instance failures, "
                "high CPU utilization, high memory utilization, "
                "instance status, server health and infrastructure "
                "health problems."
            ),
        ),
        (
            "cloud operations",
            (
                "Day-to-day cloud operations including monitoring, "
                "observability, metrics, CPU utilization, memory "
                "utilization, application health, infrastructure "
                "health, logs, alerts, incidents, outages, "
                "performance, latency, availability and production "
                "troubleshooting."
            ),
        ),
        (
            "cloud networking",
            (
                "Cloud networking concepts and operational problems "
                "including VPCs, virtual networks, subnets, route "
                "tables, routing, security groups, network ACLs, "
                "load balancers, DNS, connectivity, firewalls and "
                "network troubleshooting."
            ),
        ),
        (
            "application health",
            (
                "Investigating production applications running in "
                "cloud environments. Questions about unhealthy "
                "applications, application availability, application "
                "errors, HTTP errors, timeouts, application "
                "performance, service failures, connection problems, "
                "logs and application troubleshooting."
            ),
        ),
        (
            "incident troubleshooting",
            (
                "Investigating cloud production incidents and "
                "infrastructure failures. Troubleshooting unhealthy "
                "services, degraded systems, outages, errors, "
                "timeouts, high resource utilization, application "
                "failures, production incidents and identifying "
                "possible root causes."
            ),
        ),
        (
            "cloud deployments",
            (
                "Cloud deployment and release operations including "
                "deployment failures, recent deployments, release "
                "versions, deployment status, release pipelines, "
                "CI/CD systems, production releases and investigating "
                "whether a recent deployment may be related to an "
                "infrastructure or application incident."
            ),
        ),
        (
            "monitoring and observability",
            (
                "Cloud monitoring and observability including CPU "
                "metrics, memory metrics, infrastructure metrics, "
                "application metrics, logs, alerts, monitoring "
                "systems, performance monitoring, health checks, "
                "availability monitoring and production observability."
            ),
        ),
        (
            "devops and sre",
            (
                "DevOps and Site Reliability Engineering topics "
                "including deployments, release pipelines, CI/CD, "
                "Terraform, Ansible, Kubernetes, Docker, monitoring, "
                "reliability, production operations, incident "
                "response and site reliability engineering."
            ),
        ),
    ]

    # ============================================================
    # NEGATIVE OUT-OF-SCOPE DOMAIN PROTOTYPES
    # ============================================================

    OUT_OF_SCOPE_PROTOTYPES = [
        (
            "general programming",
            (
                "General programming and software development "
                "questions including programming languages, Java, "
                "Python programming, coding syntax, algorithms, "
                "data structures, programming tutorials and general "
                "software development unrelated to cloud operations."
            ),
        ),
        (
            "general ai and machine learning",
            (
                "General artificial intelligence, machine learning, "
                "deep learning, generative AI, Agentic AI, AI agents, "
                "large language models, neural networks and AI concepts "
                "that are not specifically about operating cloud "
                "infrastructure."
            ),
        ),
        (
            "database knowledge",
            (
                "Database-specific knowledge and database features "
                "including Oracle database versions, Oracle 21c "
                "features, SQL concepts, database architecture and "
                "database administration questions that are not "
                "specifically about cloud operations."
            ),
        ),
        (
            "general knowledge",
            (
                "General knowledge questions including geography, "
                "history, countries, capitals, politics, science and "
                "other general information unrelated to cloud "
                "infrastructure or cloud operations."
            ),
        ),
        (
            "education and exams",
            (
                "Education and examination questions including exam "
                "preparation, study plans, learning subjects, school "
                "questions, courses and educational guidance unrelated "
                "to cloud operations."
            ),
        ),
        (
            "personal questions",
            (
                "Personal questions about the user, their name, "
                "identity, personal life, preferences, opinions or "
                "private information."
            ),
        ),
        (
            "entertainment and casual conversation",
            (
                "Entertainment, jokes, poems, stories, casual "
                "conversation and general social interaction unrelated "
                "to cloud infrastructure or cloud operations."
            ),
        ),
    ]

    def __init__(
        self,
        positive_threshold: float = 0.25,
        margin_threshold: float = 0.03,
    ):
        """
        Initialize the semantic classifier.

        Args:
            positive_threshold:
                Minimum positive semantic similarity required.

            margin_threshold:
                Minimum difference required between the strongest
                positive and strongest negative semantic matches.

        These values are calibration values and should be changed
        only after evaluating the classifier against a representative
        test dataset.
        """

        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError(
                "COHERE_API_KEY environment variable "
                "is not configured."
            )

        self.client = cohere.ClientV2(
            api_key=api_key
        )

        self.positive_threshold = positive_threshold
        self.margin_threshold = margin_threshold

        self.positive_names = [
            name
            for name, _ in self.CLOUD_DOMAIN_PROTOTYPES
        ]

        self.positive_descriptions = [
            description
            for _, description in self.CLOUD_DOMAIN_PROTOTYPES
        ]

        self.negative_names = [
            name
            for name, _ in self.OUT_OF_SCOPE_PROTOTYPES
        ]

        self.negative_descriptions = [
            description
            for _, description in self.OUT_OF_SCOPE_PROTOTYPES
        ]

        self.positive_embeddings = (
            self._create_embeddings(
                self.positive_descriptions
            )
        )

        self.negative_embeddings = (
            self._create_embeddings(
                self.negative_descriptions
            )
        )

    def _create_embeddings(
        self,
        texts,
    ):
        """
        Generate embeddings for a collection of
        semantic prototype descriptions.
        """

        response = self.client.embed(
            model="embed-v4.0",
            texts=texts,
            input_type="classification",
            output_dimension=1024,
            embedding_types=["float"],
        )

        return response.embeddings.float

    def _create_question_embedding(
        self,
        question: str,
    ):
        """
        Generate an embedding for the user's question.
        """

        response = self.client.embed(
            model="embed-v4.0",
            texts=[question],
            input_type="classification",
            output_dimension=1024,
            embedding_types=["float"],
        )

        return response.embeddings.float[0]

    @staticmethod
    def _cosine_similarity(
        vector_a,
        vector_b,
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """

        dot_product = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b,
            )
        )

        magnitude_a = math.sqrt(
            sum(
                value * value
                for value in vector_a
            )
        )

        magnitude_b = math.sqrt(
            sum(
                value * value
                for value in vector_b
            )
        )

        if (
            magnitude_a == 0
            or magnitude_b == 0
        ):
            return 0.0

        return (
            dot_product
            / (magnitude_a * magnitude_b)
        )

    def _calculate_similarities(
        self,
        question_embedding,
        prototype_embeddings,
        prototype_names,
    ):
        """
        Calculate similarity between a question and
        a collection of semantic prototypes.
        """

        similarities = []

        for index, prototype_embedding in enumerate(
            prototype_embeddings
        ):
            similarity = (
                self._cosine_similarity(
                    question_embedding,
                    prototype_embedding,
                )
            )

            similarities.append(
                {
                    "domain": prototype_names[index],
                    "similarity": round(
                        similarity,
                        4,
                    ),
                }
            )

        similarities.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return similarities

    def classify(
        self,
        question: str,
    ) -> dict:
        """
        Classify a user question using positive and
        negative semantic similarity.

        Returns:

            {
                "is_cloud_operations": bool,
                "category": str,
                "confidence": float,
                "matched_domain": str,
                "positive_similarity": float,
                "negative_similarity": float,
                "margin": float,
                "positive_similarities": [...],
                "negative_similarities": [...]
            }
        """

        if not question or not question.strip():
            return {
                "is_cloud_operations": False,
                "category": "OUT_OF_SCOPE",
                "confidence": 0.0,
                "matched_domain": None,
                "positive_similarity": 0.0,
                "negative_similarity": 0.0,
                "margin": 0.0,
                "positive_similarities": [],
                "negative_similarities": [],
            }

        question_embedding = (
            self._create_question_embedding(
                question
            )
        )

        positive_similarities = (
            self._calculate_similarities(
                question_embedding,
                self.positive_embeddings,
                self.positive_names,
            )
        )

        negative_similarities = (
            self._calculate_similarities(
                question_embedding,
                self.negative_embeddings,
                self.negative_names,
            )
        )

        best_positive = positive_similarities[0]
        best_negative = negative_similarities[0]

        positive_similarity = (
            best_positive["similarity"]
        )

        negative_similarity = (
            best_negative["similarity"]
        )

        margin = round(
            positive_similarity
            - negative_similarity,
            4,
        )

        # ========================================================
        # DECISION LOGIC
        # ========================================================
        #
        # A request is considered Cloud Operations when:
        #
        # 1. Positive similarity is sufficiently strong.
        #
        # 2. The strongest positive domain is sufficiently
        #    stronger than the strongest negative domain.
        #
        # This prevents unrelated questions from being accepted
        # merely because they have some semantic relationship
        # with cloud technology.
        # ========================================================

        is_cloud_operations = (
            positive_similarity
            >= self.positive_threshold
            and margin
            >= self.margin_threshold
        )

        if is_cloud_operations:
            return {
                "is_cloud_operations": True,
                "category": "CLOUD_OPERATIONS",
                "confidence": positive_similarity,
                "matched_domain": (
                    best_positive["domain"]
                ),
                "positive_similarity": (
                    positive_similarity
                ),
                "negative_similarity": (
                    negative_similarity
                ),
                "margin": margin,
                "positive_similarities": (
                    positive_similarities
                ),
                "negative_similarities": (
                    negative_similarities
                ),
            }

        return {
            "is_cloud_operations": False,
            "category": "OUT_OF_SCOPE",
            "confidence": positive_similarity,
            "matched_domain": (
                best_positive["domain"]
            ),
            "positive_similarity": (
                positive_similarity
            ),
            "negative_similarity": (
                negative_similarity
            ),
            "margin": margin,
            "positive_similarities": (
                positive_similarities
            ),
            "negative_similarities": (
                negative_similarities
            ),
        }