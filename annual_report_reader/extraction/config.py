"""The settings of one extraction run.

Paths are relative to the repo root, so a saved run does not depend on the machine. Secrets (endpoint, API key)
stay in `.env`; loading it is the entry point's job (notebook or CLI), not this module's.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from ..chat import ReasoningEffort, Usage

# The package sits at the repo root (pyproject: module-root = ""): annual_report_reader/extraction/config.py.
REPO_ROOT = Path(__file__).resolve().parents[2]


class ExtractionConfig(BaseModel):
    """The settings of one extraction run. Frozen, and saved with the run's output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Input
    pdf_path: Path = Path("data/VestasAnnualReport2025.pdf")
    pages: tuple[PositiveInt, ...] = Field(
        (50, 51, 71, 72, 73, 74, 85, 118),
        description="1-indexed PDF pages (equal to the report's page numbers)",
    )
    dpi: PositiveInt = 200

    # Calls (each run of adjacent pages is one call, see pages.adjacent_runs)
    max_concurrency: PositiveInt = Field(4, description="Calls in flight at the same time")

    # Model and prompt
    deployment: str = Field(default_factory=lambda: os.environ["AZURE_OPENAI_DEPLOYMENT_LUNA"])
    reasoning_effort: ReasoningEffort | None = Field("high", description="None = the deployment's default")
    prompt_dir: Path = Path("prompts")
    prompt: str = "extract"
    definition_path: Path = Path("prompts/principal-risk-definition.md")

    # Cost (USD per 1M tokens)
    price_in_per_m: float = 0.20
    price_out_per_m: float = 1.20

    @field_validator("pages")
    @classmethod
    def _sorted_unique(cls, pages: tuple[int, ...]) -> tuple[int, ...]:
        if dupes := sorted({p for p in pages if pages.count(p) > 1}):
            raise ValueError(f"duplicate pages: {dupes}")
        return tuple(sorted(pages))

    def cost(self, usage: Usage) -> float:
        """USD for `usage` at this config's prices."""
        return usage.cost(self.price_in_per_m, self.price_out_per_m)

    def resolve(self, path: Path) -> Path:
        """A config path as an absolute path under the repo root."""
        return REPO_ROOT / path
