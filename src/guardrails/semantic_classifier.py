"""
Semantic Scope Classifier

Uses Cohere embeddings to determine whether a user
question belongs to the supported Cloud Operations
domain.

The classifier uses semantic domain prototypes rather
than maintaining a large keyword/question list.
"""

import math
import os

import cohere


class SemanticScopeClassifier:
    """
    Semantic classifier for the Cloud Operations Agent.

    The classifier compares a user's question against
    several representative Cloud Operations domain
    descriptions.

    It does not attempt to maintain a list of every
    possible cloud-related question.
    """

    CLOUD_DOMAIN_PROTOTYPES = [
        (
            "cloud infrastructure",
            (
                "Cloud infrastructure concepts including "
                "AWS, Amazon Web Services, GCP, Google Cloud, "
                "Microsoft Azure, EC2, virtual machines, "
                "servers, compute resources, storage, "
                "networking, VPCs, subnets, load balancers, "
                "availability and infrastructure."
            ),
        ),
        (
            "cloud platforms",
            (
                "Cloud platform technologies including "
                "AWS, GCP, Google Cloud Platform, Azure, "
                "EC2, virtual machines, containers, "
                "Kubernetes, Docker and cloud services."
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
                "troubleshooting."
            ),
        ),
        (
            "incident troubleshooting",
            (
                "Troubleshooting cloud infrastructure and "
                "production applications, investigating "
                "incidents, analyzing logs, identifying "
                "root causes, investigating failures, "
                "timeouts, errors, degraded services and "
                "unhealthy applications."
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
    ]

    def __init__(
        self,
        threshold: float = 0.30,
    ):
        """
        Initialize the semantic classifier.

        Args:
            threshold:
                Minimum cosine similarity required for a
                question to be considered cloud-related.

        The initial threshold is intentionally conservative
        but will be calibrated using our evaluation tests.
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

        self.threshold = threshold

        self.prototype_names = [
            name
            for name, _ in self.CLOUD_DOMAIN_PROTOTYPES
        ]

        self.prototype_descriptions = [
            description
            for _, description in self.CLOUD_DOMAIN_PROTOTYPES
        ]

        self.prototype_embeddings = (
            self._create_prototype_embeddings()
        )

    def _create_prototype_embeddings(self):
        """
        Generate embeddings for all Cloud Operations
        domain prototypes.
        """

        response = self.client.embed(
            model="embed-v4.0",
            texts=self.prototype_descriptions,
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

    def classify(
        self,
        question: str,
    ) -> dict:
        """
        Classify a user question using semantic
        similarity against Cloud Operations prototypes.

        Returns:

            {
                "is_cloud_operations": bool,
                "category": str,
                "confidence": float,
                "matched_domain": str
            }
        """

        if not question or not question.strip():

            return {
                "is_cloud_operations": False,
                "category": "OUT_OF_SCOPE",
                "confidence": 0.0,
                "matched_domain": None,
            }

        question_embedding = (
            self._create_question_embedding(
                question
            )
        )

        similarities = []

        for index, prototype_embedding in enumerate(
            self.prototype_embeddings
        ):

            similarity = (
                self._cosine_similarity(
                    question_embedding,
                    prototype_embedding,
                )
            )

            similarities.append(
                {
                    "domain": self.prototype_names[index],
                    "similarity": similarity,
                }
            )

        # Find the strongest semantic match.
        best_match = max(
            similarities,
            key=lambda item: item["similarity"],
        )

        best_similarity = (
            best_match["similarity"]
        )

        is_cloud_operations = (
            best_similarity >= self.threshold
        )

        if is_cloud_operations:

            return {
                "is_cloud_operations": True,
                "category": "CLOUD_OPERATIONS",
                "confidence": round(
                    best_similarity,
                    4,
                ),
                "matched_domain": (
                    best_match["domain"]
                ),
            }

        return {
            "is_cloud_operations": False,
            "category": "OUT_OF_SCOPE",
            "confidence": round(
                best_similarity,
                4,
            ),
            "matched_domain": (
                best_match["domain"]
            ),
        }