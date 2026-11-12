"""The merge prompt: a system prompt file plus one user message with the mentions as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from .mentions import Mention


def load_merge_prompt(prompt_dir: Path, name: str) -> ChatPromptTemplate:
    """The merge prompt `name`: the system file, then the mentions as JSON in the user message."""
    system = (prompt_dir / f"{name}.md").read_text(encoding="utf-8").removesuffix("\n")
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{mentions}")])
    if prompt.input_variables != ["mentions"]:  # e.g. a stray {placeholder} in the file
        raise ValueError(f"{name}: unexpected template variables {prompt.input_variables}")
    return prompt


def mentions_payload(mentions: list[Mention]) -> str:
    """The user message: every mention with its id, as a JSON list."""
    return json.dumps([m.model_dump(mode="json") for m in mentions], ensure_ascii=False, indent=1)
