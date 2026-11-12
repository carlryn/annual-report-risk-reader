"""The golden sets: step 1, the principal risks on each labelled page (eval/golden/risk_extraction.yaml); step 2,
every unique risk of the report once with all its locations (eval/golden/merging_and_summarizing.yaml).

Step 1 has one entry per risk per page: a risk mentioned on two pages is expected on both. A page listed as `[]` is labelled
and has no principal risks. Categories are the extraction's own `Category` and unknown keys are rejected, so a typo
in the file fails on load instead of silently never matching.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from ..extraction.schema import Category
from ..merge.schema import Location


class GoldenRisk(BaseModel):
    """One labelled principal risk on one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page: int
    title: str
    category: Category
    # `secondary` in the YAML files; named like the extraction's field everywhere in code
    secondary_category: Category | None = Field(None, validation_alias=AliasChoices("secondary_category", "secondary"))
    section: str
    description: str
    mitigation: str  # "None" when the page states none


Golden = dict[int, list[GoldenRisk]]  # labelled page -> its risks ([] = none)


def load_golden(path: Path) -> Golden:
    """Golden risks per labelled page, in page order."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {int(page): [GoldenRisk(page=int(page), **r) for r in risks or []] for page, risks in sorted(data.items())}


class GoldenRecord(BaseModel):
    """One labelled unique risk of the report (step 2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    category: Category
    secondary_category: Category | None = Field(None, validation_alias=AliasChoices("secondary_category", "secondary"))
    locations: list[Location] = Field(min_length=1)  # most informative first
    description: str
    mitigation: str  # "None" when the report states none


def load_golden_records(path: Path) -> list[GoldenRecord]:
    """The step-2 golden records, in file order."""
    return [GoldenRecord(**r) for r in yaml.safe_load(path.read_text(encoding="utf-8")) or []]
