"""The merge chain (chat model + strict structured output) and the one call per report."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from ..chat import Usage
from .config import MergeConfig
from .mentions import Mention
from .prompts import mentions_payload
from .schema import MergedRisk, MergedRisks


def merge_chain(prompt: ChatPromptTemplate, model: BaseChatModel) -> Runnable:
    """list[Mention] -> {"raw": AIMessage, "parsed": MergedRisks | None, "parsing_error": Exception | None}."""
    structured = model.with_structured_output(MergedRisks, method="json_schema", strict=True, include_raw=True)
    return RunnableLambda(lambda mentions: {"mentions": mentions_payload(mentions)}) | prompt | structured


class MergeCallResult(BaseModel):
    """The merge call: the unique risks that came back, or the error."""

    risks: list[MergedRisk] = []
    usage: Usage = Usage()
    error: str | None = None


def call_result(output: dict[str, Any] | Exception) -> MergeCallResult:
    """The chain's output (or the exception it raised, after retries) as a MergeCallResult."""
    if isinstance(output, Exception):
        return MergeCallResult(error=repr(output))
    usage = Usage.of(output["raw"])
    if output["parsed"] is None:  # a refusal, or a response that does not fit the schema
        return MergeCallResult(usage=usage, error=f"no parsed output: {output['parsing_error']!r}")
    return MergeCallResult(risks=output["parsed"].risks, usage=usage)


def merge_mentions(chain: Runnable, mentions: list[Mention], config: MergeConfig, run_id: str) -> MergeCallResult:
    """One call for the whole report; a failed call comes back as a MergeCallResult with an error."""
    run_config = {
        "run_name": "merge",
        "metadata": {"run_id": run_id},
    }
    try:
        output = chain.invoke(mentions, config=run_config)
    except Exception as e:  # noqa: BLE001 - a failed call is saved as a failed run
        output = e
    return call_result(output)
