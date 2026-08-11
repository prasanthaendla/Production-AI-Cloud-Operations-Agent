"""
Cohere LLM Client

Handles communication with the Cohere API.
"""

import os

import cohere
from dotenv import load_dotenv


load_dotenv()


class CohereClient:
    """
    Wrapper around the Cohere Chat API.
    """

    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")

        if not api_key:
            raise ValueError(
                "COHERE_API_KEY is not configured."
            )

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
        Send a chat request to Cohere.

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