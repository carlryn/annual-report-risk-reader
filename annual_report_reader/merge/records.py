"""The final records from the model's answer, with its mention ids checked in code.

A JSON schema cannot say "every id must be a real mention, used once", so code does: an unknown or already used id
is dropped (first use wins), a risk with no valid id left is dropped, and a mention the model left out becomes its
own record, copied as is, so nothing step 1 found is silently lost. Each fix is kept as a `Problem`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .mentions import Mention
from .schema import Location, MergedRisk, RiskRecord


class Problem(BaseModel):
    """Something the code had to fix in the model's answer."""

    kind: Literal["unknown id", "repeated id", "no valid ids", "left out"]
    risk_title: str  # the model's risk, or the mention's title for "left out"
    mention_id: str | None = None


def build_records(merged: list[MergedRisk], mentions: list[Mention]) -> tuple[list[RiskRecord], list[Problem]]:
    """The final records from the model's unique risks, with ids checked and every mention accounted for."""
    by_id = {m.mention_id: m for m in mentions}
    used: set[str] = set()
    records: list[RiskRecord] = []
    problems: list[Problem] = []

    def add(ids: list[str], **fields) -> None:
        locations = [Location(page=by_id[i].page, section=by_id[i].section) for i in ids]
        records.append(RiskRecord(id=f"r{len(records):02d}", locations=locations, mention_ids=ids, **fields))

    for risk in merged:
        ids: list[str] = []
        for i in risk.mention_ids:
            if i not in by_id:
                problems.append(Problem(kind="unknown id", risk_title=risk.title, mention_id=i))
            elif i in used or i in ids:
                problems.append(Problem(kind="repeated id", risk_title=risk.title, mention_id=i))
            else:
                ids.append(i)
        if not ids:
            problems.append(Problem(kind="no valid ids", risk_title=risk.title))
            continue
        used.update(ids)
        add(ids, **risk.model_dump(exclude={"mention_ids"}))

    for m in mentions:
        if m.mention_id not in used:
            problems.append(Problem(kind="left out", risk_title=m.title, mention_id=m.mention_id))
            fields = m.model_dump(include={"title", "category", "secondary_category", "description", "mitigation"})
            add([m.mention_id], **fields)
    return records, problems
