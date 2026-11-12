"""LangChain chat models and token usage, shared by every LLM step (extraction, evaluation, ...).

Endpoint and key come from the environment (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY); loading `.env` is the
entry point's job.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

API_VERSION = "2025-04-01-preview"

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


def azure_chat_model(deployment: str, reasoning_effort: ReasoningEffort | None) -> AzureChatOpenAI:
    """An Azure OpenAI deployment as a LangChain chat model."""
    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", API_VERSION),
        reasoning_effort=reasoning_effort,
        max_retries=3,  # rate limits and 5xx, with exponential backoff
        timeout=30,
    )


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # included in output_tokens

    @classmethod
    def of(cls, message: AIMessage) -> Usage:
        u = message.usage_metadata or {}
        return cls(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            reasoning_tokens=(u.get("output_token_details") or {}).get("reasoning", 0),
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{f: getattr(self, f) + getattr(other, f) for f in Usage.model_fields})

    def cost(self, price_in_per_m: float, price_out_per_m: float) -> float:
        """USD, at the given prices per 1M tokens."""
        return (self.input_tokens * price_in_per_m + self.output_tokens * price_out_per_m) / 1e6
