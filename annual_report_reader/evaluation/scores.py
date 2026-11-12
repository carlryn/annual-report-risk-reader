"""Recall, precision and F1 (per page and overall for step 1; per report for step 2), and the share of yes grades.

Overall scores are micro-averaged (counts summed over pages). A page with nothing on one side has no recall or
precision (None), but still counts towards the totals. Pages whose judge failed are left out.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, computed_field

from .judge import PageResult


class Counts(BaseModel):
    golden: int = 0
    predicted: int = 0
    paired: int = 0

    @classmethod
    def of(cls, r: PageResult) -> Counts:
        return cls(golden=len(r.golden), predicted=len(r.predicted), paired=len(r.pairs))

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            golden=self.golden + other.golden,
            predicted=self.predicted + other.predicted,
            paired=self.paired + other.paired,
        )

    @computed_field
    @property
    def recall(self) -> float | None:
        return self.paired / self.golden if self.golden else None

    @computed_field
    @property
    def precision(self) -> float | None:
        return self.paired / self.predicted if self.predicted else None

    @computed_field
    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0


def scored(results: list[PageResult]) -> list[PageResult]:
    """The pages that count: every page whose judge did not fail."""
    return [r for r in results if r.judge_error is None]


def total(results: list[PageResult]) -> Counts:
    return sum((Counts.of(r) for r in scored(results)), Counts())


def scores_table(results: list[PageResult]) -> pd.DataFrame:
    """Recall, precision and F1 per scored page, and overall (micro)."""
    rows = {f"p.{r.page}": Counts.of(r) for r in scored(results)} | {"all": total(results)}
    return pd.DataFrame.from_dict({k: c.model_dump() for k, c in rows.items()}, orient="index")


def share_yes(items: list, fields: list[str] | tuple[str, ...]) -> dict[str, float | None]:
    """Per field, the share of items graded "yes" (or True); None when there are no items."""
    return {f: (sum(getattr(i, f) in ("yes", True) for i in items) / len(items) if items else None) for f in fields}
