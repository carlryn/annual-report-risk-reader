"""Rules every merged record must follow, checked in code without a golden or a judge.

- the description has at most 3 sentences (the merge prompt asks for 2, at most 3);
- the mitigation is "None" exactly when none of the record's mentions states one (none invented, none dropped);
- the title has at most 10 words (the prompt asks for about 8).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, computed_field

from ..merge.mentions import Mention
from ..merge.schema import RiskRecord

CHECKS = ("description_max_3_sentences", "mitigation_consistent", "title_max_10_words")


def n_sentences(text: str) -> int:
    """Sentences split on . ! ? followed by a space and a capital, so "e.g." and "6.39" do not split."""
    return len([s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if s])


def states_none(text: str) -> bool:
    return text.strip().lower().rstrip(".") in {"none", "none stated", ""}


class RecordChecks(BaseModel):
    """The code checks of one record."""

    record_id: str
    sentences: int
    description_max_3_sentences: bool
    mitigation_consistent: bool
    title_max_10_words: bool

    @computed_field
    @property
    def passed(self) -> bool:
        return all(getattr(self, c) for c in CHECKS)


def check_record(record: RiskRecord, mentions: list[Mention]) -> RecordChecks:
    by_id = {m.mention_id: m for m in mentions}
    stated_in_source = any(not states_none(by_id[i].mitigation) for i in record.mention_ids)
    sentences = n_sentences(record.description)
    return RecordChecks(
        record_id=record.id,
        sentences=sentences,
        description_max_3_sentences=sentences <= 3,
        mitigation_consistent=stated_in_source == (not states_none(record.mitigation)),
        title_max_10_words=len(record.title.split()) <= 10,
    )
