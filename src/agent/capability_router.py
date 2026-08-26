"""
Capability Router

Determines whether a cloud-related user question
matches a capability supported by the AI Cloud
Operations Agent.

The router uses semantic similarity plus a confidence
margin between the strongest and second-strongest
capability.

Stage 18:
- Centralized configuration.
- Retry handling for transient Cohere embedding failures.
- Exponential backoff.
- Optional client injection for testing.
- Lightweight intent fallback for clear EC2
  instance health/status questions.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

import cohere
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agent.capabilities import (
    get_capability_descriptions,
)
from src.config import settings

from cohere.errors import (
    InternalServerError,
    ServiceUnavailableError,
    TooManyRequestsError,
)


class CapabilityRouter:
    """
    Semantic router for supported agent capabilities.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        margin_threshold: float = 0.03,
        client: Optional[Any] = None,
    ):
        """
        Initialize the capability router.

        Args:
            threshold:
                Minimum similarity required for a
                capability match.

            margin_threshold:
                Minimum separation required between
                the best capability and the second-best
                capability.

            client:
                Optional Cohere client.

                Production:
                    Leave as None.

                Tests:
                    Inject a mock client.
        """

        if client is None:

            settings.validate()

            self.client = cohere.ClientV2(
                api_key=settings.cohere_api_key
            )

        else:
            self.client = client

        self.threshold = threshold

        self.margin_threshold = (
            margin_threshold
        )

        self.capability_descriptions = (
            get_capability_descriptions()
        )

        self.capability_names = list(
            self.capability_descriptions.keys()
        )

        self.prototype_descriptions = list(
            self.capability_descriptions.values()
        )

        self.prototype_embeddings = (
            self._create_prototype_embeddings()
        )

    # ==============================================================
    # EMBEDDINGS
    # ==============================================================

    @retry(
        retry=retry_if_exception_type(
            (
                TooManyRequestsError,
                InternalServerError,
                ServiceUnavailableError,
            )
        ),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _embed(
        self,
        texts,
    ):
        """
        Generate Cohere embeddings with retry handling.

        Retries transient API failures:

        - HTTP 429 Too Many Requests
        - HTTP 500 Internal Server Error
        - HTTP 503 Service Unavailable
        """

        return self.client.embed(
            model="embed-v4.0",
            texts=texts,
            input_type="classification",
            output_dimension=1024,
            embedding_types=["float"],
        )

    def _create_prototype_embeddings(self):
        """
        Create embeddings for capability descriptions.
        """

        response = self._embed(
            self.prototype_descriptions
        )

        return response.embeddings.float

    def _create_question_embedding(
        self,
        question: str,
    ):
        """
        Create an embedding for the user's question.
        """

        response = self._embed(
            [question]
        )

        return response.embeddings.float[0]

    # ==============================================================
    # SIMILARITY
    # ==============================================================

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
            / (
                magnitude_a
                * magnitude_b
            )
        )

    # ==============================================================
    # INSTANCE HEALTH INTENT
    # ==============================================================

    @staticmethod
    def _is_instance_health_question(
        question: str,
    ) -> bool:
        """
        Detect clear EC2 instance health/status questions.

        This is intentionally lightweight.

        The semantic router remains the primary routing
        mechanism. This fallback only handles obvious
        instance-health questions that may receive a
        low semantic margin because of wording differences.

        Examples:

            Why is instance i-123 unhealthy?
            What is the status of EC2 instance i-123?
            What is the health of instance i-123?
            Is EC2 instance i-123 healthy?
            Check the status of instance i-123.
        """

        normalized = question.lower().strip()

        # ----------------------------------------------------------
        # Require an EC2/instance reference
        # ----------------------------------------------------------

        has_instance_reference = bool(
            re.search(
                r"\bi-[a-z0-9]+\b",
                normalized,
            )
        )

        if not has_instance_reference:

            return False

        has_ec2_reference = (
            "ec2" in normalized
            or "instance" in normalized
        )

        if not has_ec2_reference:

            return False

        # ----------------------------------------------------------
        # Health/status intent
        # ----------------------------------------------------------

        health_terms = (
            "health",
            "healthy",
            "unhealthy",
            "status",
            "running",
            "stopped",
            "failed",
            "failure",
            "impaired",
            "degraded",
            "check",
            "checks",
        )

        return any(
            term in normalized
            for term in health_terms
        )

    # ==============================================================
    # ROUTING
    # ==============================================================

    def route(
        self,
        question: str,
    ) -> dict:
        """
        Determine the best matching capability.

        Returns:

            {
                "is_supported": bool,
                "capability": str | None,
                "confidence": float,
                "margin": float,
            }

        The semantic router is used first.

        A lightweight fallback is then used for
        unambiguous EC2 instance-health/status questions
        when semantic similarity alone is not sufficiently
        separated.
        """

        if (
            not question
            or not question.strip()
        ):
            return {
                "is_supported": False,
                "capability": None,
                "confidence": 0.0,
                "margin": 0.0,
            }

        # ----------------------------------------------------------
        # Semantic routing
        # ----------------------------------------------------------

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
                    "capability": (
                        self.capability_names[index]
                    ),
                    "similarity": similarity,
                }
            )

        similarities.sort(
            key=lambda item: item[
                "similarity"
            ],
            reverse=True,
        )

        best_match = similarities[0]

        best_similarity = (
            best_match["similarity"]
        )

        if len(similarities) > 1:

            second_similarity = (
                similarities[1]["similarity"]
            )

        else:

            second_similarity = 0.0

        margin = (
            best_similarity
            - second_similarity
        )

        is_supported = (
            best_similarity >= self.threshold
            and margin >= self.margin_threshold
        )

        if is_supported:

            return {
                "is_supported": True,
                "capability": (
                    best_match["capability"]
                ),
                "confidence": round(
                    best_similarity,
                    4,
                ),
                "margin": round(
                    margin,
                    4,
                ),
            }

        # ----------------------------------------------------------
        # Lightweight EC2 instance-health fallback
        # ----------------------------------------------------------
        #
        # If the semantic router cannot confidently separate
        # capabilities but the question is clearly asking about
        # the health/status of a specific EC2 instance, route it
        # to the existing instance_health capability.
        #

        if (
            "instance_health"
            in self.capability_names
            and self._is_instance_health_question(
                question
            )
        ):

            return {
                "is_supported": True,
                "capability": "instance_health",
                "confidence": round(
                    best_similarity,
                    4,
                ),
                "margin": round(
                    margin,
                    4,
                ),
            }

        # ----------------------------------------------------------
        # Unsupported
        # ----------------------------------------------------------

        return {
            "is_supported": False,
            "capability": None,
            "confidence": round(
                best_similarity,
                4,
            ),
            "margin": round(
                margin,
                4,
            ),
        }