"""
Capability Router

Determines whether a cloud-related user question
matches a capability supported by the AI Cloud
Operations Agent.

The router uses semantic similarity plus a confidence
margin between the strongest and second-strongest
capability.
"""

import math
from typing import Any, Optional

import cohere

from src.agent.capabilities import (
    get_capability_descriptions,
)
from src.config import settings


class CapabilityRouter:
    """
    Semantic router for supported agent capabilities.

    Production:
        Uses the real Cohere client.

    Testing:
        A mock/injected client can be supplied so that
        unit tests do not call the Cohere API.
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
                Optional embedding client.

                Production:
                    Leave as None and the router creates
                    the real Cohere client.

                Tests:
                    Inject a mock client to avoid external
                    API calls.
        """

        if client is None:

            settings.validate()

            api_key = settings.cohere_api_key

            self.client = cohere.ClientV2(
                api_key=api_key
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

    def _create_prototype_embeddings(self):
        """
        Create embeddings for capability descriptions.
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
        Create an embedding for the user's question.
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
            / (
                magnitude_a
                * magnitude_b
            )
        )

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