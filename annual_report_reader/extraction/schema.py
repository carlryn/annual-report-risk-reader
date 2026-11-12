"""What one extraction call returns. Fields match the golden set (eval/golden/risk_extraction.yaml).

Sent as strict structured output: every field is required, and "no value" is an explicit null
(or "None" for mitigation). The model sees the docstrings and field descriptions, so they hold the per-field
rules; prompts/extract.md holds the task and the rules that span fields. Keep each rule in one place.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Category(StrEnum):
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"
    MARKET = "market"
    CLIMATE = "climate"
    CYBER = "cyber"
    SUPPLY_CHAIN = "supply chain"
    OTHER = "other"


class Confidence(StrEnum):
    LOW = "low"
    HIGH = "high"


class Risk(BaseModel):
    """One principal risk, as it appears on one page."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="The page number given before the page's image, not the number printed on the page.")
    title: str = Field(description="A short title. Use the title printed on the page if there is one.")
    category: Category = Field(description="The allowed category that fits best.")
    secondary_category: Category | None = Field(
        description="A second allowed category only if it also clearly fits; otherwise null."
    )
    section: str = Field(
        description="The heading the risk sits under, as printed. If the risk is in a table, add the column or row "
        'heading too, e.g. "Principal risks - Operational risks".'
    )
    description: str = Field(
        description="The description from the page, copied word for word. If it is longer than 15 sentences, keep "
        "the 15 most informative ones. Do not summarise or add anything."
    )
    mitigation: str = Field(
        description='The mitigation actions stated for this risk, copied word for word. "None" if none are stated. '
        "Do not infer mitigations."
    )
    confidence: Confidence = Field(
        description="high: clearly meets every characteristic of a principal risk. low: borderline on one or more, "
        "e.g. close to an impact, an opportunity or a generic statement."
    )


class ExtractedRisks(BaseModel):
    """What one call returns: the principal risks on a group of adjacent pages."""

    model_config = ConfigDict(extra="forbid")

    risks: list[Risk]
