"""
Cohere LLM Client

Handles communication with the Cohere API.

Stage 18:
- Centralized configuration.
- Retry handling for transient API failures.
- Exponential backoff.
- Controlled failure after retry exhaustion.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import cohere
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings


# Cohere SDK exception classes are imported directly.
#
# Cohere 7.x does not expose these through:
#
#     cohere.errors.SomeError
#
# Therefore they must be imported from the SDK's
# exception module.
from cohere.errors import (
    InternalServerError,
    ServiceUnavailableError,
    TooManyRequestsError,
)


class CohereClient:
    """
    Production-safe wrapper around the Cohere Chat API.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
    ):
        """
        Initialize the Cohere client.

        Args:
            client:
                Optional client injection for testing.

                Production:
                    Leave as None.

                Tests:
                    Inject a mock client.
        """

        settings.validate()

        self.client = (
            client
            if client is not None
            else cohere.ClientV2(
                api_key=settings.cohere_api_key
            )
        )

        self.model = "command-a-03-2025"

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
    def _chat_with_retry(
        self,
        messages: Iterable[Any],
        tools=None,
    ):
        """
        Execute a Cohere chat request with retry handling.

        Retries are limited to transient failures such as:

        - HTTP 429 Too Many Requests
        - HTTP 500 Internal Server Error
        - HTTP 503 Service Unavailable

        Non-transient errors are allowed to propagate
        immediately.
        """

        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        return self.client.chat(**kwargs)

    def chat(
        self,
        messages,
        tools=None,
    ):
        """
        Execute a Cohere chat request.

        Returns:
            Cohere response.

        Raises:
            Cohere API exception after the configured
            retry policy is exhausted.
        """

        return self._chat_with_retry(
            messages=messages,
            tools=tools,
        )