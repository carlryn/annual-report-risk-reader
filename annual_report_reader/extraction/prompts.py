"""The extraction prompt: a system prompt file plus one message holding the page images."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .pages import PageGroup


def load_prompt(prompt_dir: Path, name: str, definition: str) -> ChatPromptTemplate:
    """The extraction prompt `name`, with the principal-risk definition filled in. Its only input is `pages`."""
    system = (prompt_dir / f"{name}.md").read_text(encoding="utf-8").removesuffix("\n")
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder("pages")])
    prompt = prompt.partial(PRINCIPAL_RISK_DEFINITION=definition.removesuffix("\n"))
    if prompt.input_variables != ["pages"]:  # e.g. a stray {placeholder} in the file
        raise ValueError(f"{name}: unexpected template variables {prompt.input_variables}")
    return prompt


def pages_message(group: PageGroup) -> HumanMessage:
    """One user message with every page of the group: "Page N", then its image."""
    content: list[str | dict] = []
    for image in group.images:
        content.append({"type": "text", "text": f"Page {image.page}"})
        content.append({"type": "image_url", "image_url": {"url": image.data_url}})
    return HumanMessage(content=content)
