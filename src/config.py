"""
Application configuration.

Centralizes environment-based configuration and
keeps secrets outside the source code.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application runtime configuration."""

    def __init__(self) -> None:
        self.cohere_api_key = os.getenv("COHERE_API_KEY")

    def validate(self) -> None:
        """Validate required production configuration."""

        if not self.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is not configured."
            )


settings = Settings()
