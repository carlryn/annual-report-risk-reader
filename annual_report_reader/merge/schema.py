"""What the merge call returns (`MergedRisks`) and the final record built from it (`RiskRecord`).

The model groups mentions and writes the text; code adds the record id and copies each location from its mention,
so pages and sections are always what step 1 extracted. Fields match eval/golden/merging_and_summarizing.yaml.

The model sees the docstrings and field descriptions of `MergedRisks`, so they hold the per-field rules;
prompts/merge.md holds the task and the rules for grouping mentions. Keep each rule in one place.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..extraction.schema import Category


class MergedRisk(BaseModel):
    """One unique risk, as the model returns it."""

    model_config = ConfigDict(extra="forbid")

    mention_ids: list[str] = Field(
        description="The ids of all mentions of this risk, ordered by relevance: the mention with the most "
        "information about the risk (description and mitigation) comes first."
    )
    title: str = Field(
        description="A short title of at most about 8 words, in sentence case (capitalise only the first word and "
        "proper nouns). Use the report's own name for the risk where possible. Leave out qualifiers such as "
        '"(entity-specific)".'
    )
    category: Category = Field(description="The allowed category that fits best, judged from all mentions.")
    secondary_category: Category | None = Field(
        description="A second allowed category only if it also clearly fits; otherwise null."
    )
    description: str = Field(
        description="What the risk is, what drives it, and how it could affect the company, combining the "
        "information from all mentions. At most 3 sentences, and prefer 2. If the mentions together contain only "
        "one sentence of description, one sentence is fine: do not pad it. Use only facts stated in the mentions; "
        "do not add numbers or claims."
    )
    mitigation: str = Field(
        description="A summary of the mitigation actions stated in the mentions, at most 3 sentences. "
        '"None" if no mention states a mitigation. Do not infer or invent mitigations.'
    )


class MergedRisks(BaseModel):
    """What the merge call returns: every unique risk of the report."""

    model_config = ConfigDict(extra="forbid")

    risks: list[MergedRisk]


class Location(BaseModel):
    """Where one mention of a risk is: copied from the mention, never written by the model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page: int
    section: str


class RiskRecord(BaseModel):
    """One unique principal risk: the output of the pipeline."""

    model_config = ConfigDict(extra="forbid")

    id: str  # r00, r01, ... in output order
    title: str
    category: Category
    secondary_category: Category | None
    locations: list[Location]  # most informative mention first
    description: str
    mitigation: str
    mention_ids: list[str]  # the step-1 mentions it was built from, in the order of `locations`
