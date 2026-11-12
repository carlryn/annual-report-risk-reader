"""The extraction chain (chat model + strict structured output) and running it over page groups."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from ..chat import Usage
from .config import ExtractionConfig
from .pages import PageGroup
from .prompts import pages_message
from .schema import ExtractedRisks, Risk


def extraction_chain(prompt: ChatPromptTemplate, model: BaseChatModel) -> Runnable:
    """PageGroup -> {"raw": AIMessage, "parsed": ExtractedRisks | None, "parsing_error": Exception | None}."""
    structured = model.with_structured_output(ExtractedRisks, method="json_schema", strict=True, include_raw=True)
    return RunnableLambda(lambda group: {"pages": [pages_message(group)]}) | prompt | structured


class CallResult(BaseModel):
    """One call: the pages sent, and the risks that came back, or the error."""

    pages: list[int]
    risks: list[Risk] = []
    usage: Usage = Usage()
    error: str | None = None


def call_result(group: PageGroup, output: dict[str, Any] | Exception) -> CallResult:
    """The chain's output (or the exception it raised, after retries) as a CallResult."""
    if isinstance(output, Exception):
        return CallResult(pages=group.pages, error=repr(output))
    usage = Usage.of(output["raw"])
    if output["parsed"] is None:  # a refusal, or a response that does not fit the schema
        return CallResult(pages=group.pages, usage=usage, error=f"no parsed output: {output['parsing_error']!r}")
    return CallResult(pages=group.pages, risks=output["parsed"].risks, usage=usage)


def extract_groups(chain: Runnable, groups: list[PageGroup], config: ExtractionConfig, run_id: str) -> list[CallResult]:
    """Run every group concurrently; failed calls come back as CallResults with an error."""
    run_configs = [
        {
            "run_name": f"extract {g.label}",
            "max_concurrency": config.max_concurrency,
            "metadata": {"run_id": run_id, "pages": g.pages},
        }
        for g in groups
    ]
    outputs = chain.batch(groups, config=run_configs, return_exceptions=True)
    return [call_result(g, out) for g, out in zip(groups, outputs, strict=True)]
