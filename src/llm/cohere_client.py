"""
Cohere LLM Client

Handles communication with the Cohere API.
"""

import cohere

from src.config import settings


class CohereClient:
    """
    Wrapper around the Cohere Chat API.
    """

    def __init__(self):
        settings.validate()

        api_key = settings.cohere_api_key

        self.client = cohere.ClientV2(
            api_key=api_key
        )

        self.model = "command-a-03-2025"

    def chat(
        self,
        messages,
        tools=None,
    ):
        """
        Args:
            messages: Conversation messages.
            tools: Optional tool definitions.

        Returns:
            Cohere response.
        """

        kwargs = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = tools

        return self.client.chat(**kwargs)