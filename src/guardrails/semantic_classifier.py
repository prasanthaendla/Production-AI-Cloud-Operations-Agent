"""
Semantic Scope Classifier

Uses Cohere embeddings to determine whether a user
question belongs to the supported Cloud Operations
domain.

The classifier uses both positive and negative
semantic domain prototypes.

Positive prototypes describe Cloud Operations.

Negative prototypes describe common domains that
should remain outside the Cloud Operations Agent.

The classifier does not maintain a list of individual
keywords or questions.
"""

import math
import os

import cohere


class SemanticScopeClassifier:
    """
    Semantic classifier for the Cloud Operations Agent.

    The classifier compares a user's question against:

    1. Positive Cloud Operations prototypes.
    2. Negative out-of-scope prototypes.

    The final decision uses:

    - Positive semantic similarity.
    - Negative semantic similarity.
    - Semantic margin between positive and negative scores.
    """

    # ============================================================
    # POSITIVE DOMAIN PROTOTYPES
    # ============================================================

    CLOUD_DOMAIN_PROTOTYPES = [
        (
            "cloud infrastructure",
            (
                "Cloud infrastructure concepts including "
                "AWS, Amazon Web Services, GCP, Google Cloud, "
                "Microsoft Azure, EC2, virtual machines, "
                "servers, compute resources, storage, "
                "networking, VPCs, subnets, route tables, "
                "load balancers, availability and "
                "cloud infrastructure."
            ),
        ),
        (
            "cloud operations",
            (
                "Cloud operations activities including "
                "monitoring, observability, metrics, CPU, "
                "memory, application health, infrastructure "
                "health, logs, alerts, incidents, outages, "
                "performance, latency, availability and "
                "troubleshooting of cloud environments."
            ),
        ),
        (
            "cloud networking",
            (
                "Cloud networking concepts including "
                "VPCs, virtual networks, subnets, route "
                "tables, routing, security groups, network "
                "ACLs, load balancers, DNS, connectivity, "
                "firewalls and network troubleshooting."
            ),
        ),
        (
            "incident troubleshooting",
            (
                "Troubleshooting cloud infrastructure and "
                "production applications, investigating "
                "incidents, analyzing logs, identifying "
                "root causes, investigating failures, "
                "timeouts, errors, degraded services, "
                "unhealthy applications and outages."
            ),
        ),
        (
            "devops and sre",
            (
                "DevOps and SRE topics including deployments, "
                "release pipelines, CI/CD, Terraform, "
                "Ansible, Kubernetes, Docker, monitoring, "
                "reliability, production operations and "
                "site reliability engineering."
            ),
        ),
        (
            "cloud applications",
            (
                "Production applications running in cloud "
                "environments, application availability, "
                "application health, application errors, "
                "application performance, service failures, "
                "timeouts, HTTP errors, logs and operational "
                "troubleshooting."
            ),
        ),
    ]

    # ============================================================
    # NEGATIVE DOMAIN PROTOTYPES
    # ============================================================

    OUT_OF_SCOPE_PROTOTYPES = [
        (
            "general programming",
            (
                "General programming and software development "
                "questions including programming languages, "
                "learning Java, Python programming, coding "
                "syntax, algorithms, data structures and "
                "general software development unrelated to "
                "cloud operations."
            ),
        ),
        (
            "general ai and machine learning",
            (
                "General artificial intelligence, machine "
                "learning, deep learning, generative AI, "
                "large language models, neural networks and "
                "AI concepts that are not specifically about "
                "operating cloud infrastructure."
            ),
        ),
        (
            "database knowledge",
            (
                "Database-specific knowledge and database "
                "features including Oracle database versions, "
                "Oracle 21c features, SQL concepts, database "
                "architecture and database administration "
                "questions that are not specifically about "
                "cloud operations."
            ),
        ),
        (
            "general knowledge",
            (
                "General knowledge questions including "
                "geography, history, countries, capitals, "
                "politics, science and other general "
                "information unrelated to cloud operations."
            ),
        ),
        (
            "education and exams",
            (
                "Education and examination questions including "
                "exam preparation, study plans, learning "
                "subjects, school questions and educational "
                "guidance unrelated to cloud operations."
            ),
        ),
        (
            "personal questions",
            (
                "Personal questions about the user, their name, "
                "identity, personal life, preferences or "
                "private information."
            ),
        ),
        (
            "entertainment and casual conversation",
            (
                "Entertainment, jokes, poems, stories, casual "
                "conversation and general social interaction "
                "unrelated to cloud infrastructure or "
                "cloud operations."
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
                Minimum positive similarity required.

            margin_threshold:
                Minimum difference required between the
                strongest positive and strongest negative
                semantic matches.

        The values are initial calibration values based
        on the classifier diagnostic evaluation dataset.
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

        self.positive_threshold = (
            positive_threshold
        )

        self.margin_threshold = (
            margin_threshold
        )

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

        best_positive = (
            positive_similarities[0]
        )

        best_negative = (
            negative_similarities[0]
        )

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
        # 1. There is sufficient positive semantic similarity.
        #
        # 2. The strongest positive domain is sufficiently
        #    stronger than the strongest negative domain.
        #
        # This relative margin is more robust than relying
        # on a fixed negative similarity threshold.
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