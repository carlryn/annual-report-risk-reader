"""Mentions: the risks of one extraction run, each with an id the merge model refers to."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict

from ..extraction.runs import ExtractionRun
from ..extraction.schema import Category


class Mention(BaseModel):
    """One step-1 risk as the merge sees it (confidence is left out: the merge prompt does not use it)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mention_id: str
    page: int
    title: str
    category: Category
    secondary_category: Category | None
    section: str
    description: str
    mitigation: str


def mentions_from_run(run: ExtractionRun) -> list[Mention]:
    """Every risk of the run in page order (run order within a page), with id p<page>-<position>."""
    position: Counter[int] = Counter()
    mentions = []
    for risk in sorted(run.risks, key=lambda r: r.page):  # stable: keeps run order within a page
        mention_id = f"p{risk.page}-{position[risk.page]}"
        mentions.append(Mention(mention_id=mention_id, **risk.model_dump(exclude={"confidence"})))
        position[risk.page] += 1
    return mentions
