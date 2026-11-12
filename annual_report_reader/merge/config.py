"""The settings of one merge run.

The input is an extraction run named by id: there is no "latest", because the newest run is often an experiment
(another model, another prompt), and a merge must never build on one by accident. Loading `.env` is the entry
point's job.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..chat import ReasoningEffort, Usage
from ..extraction.config import REPO_ROOT


class MergeConfig(BaseModel):
    """The settings of one merge run. Frozen, and saved with the run's output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Input
    extraction_run: str = Field(description="Run id in runs/extract/")

    # Model and prompt
    deployment: str = Field(default_factory=lambda: os.environ["AZURE_OPENAI_DEPLOYMENT_LUNA"])
    reasoning_effort: ReasoningEffort | None = Field("medium", description="None = the deployment's default")
    prompt_dir: Path = Path("prompts")
    prompt: str = "merge"

    # Cost (USD per 1M tokens)
    price_in_per_m: float = 0.20
    price_out_per_m: float = 1.20

    def cost(self, usage: Usage) -> float:
        """USD for `usage` at this config's prices."""
        return usage.cost(self.price_in_per_m, self.price_out_per_m)

    def resolve(self, path: Path) -> Path:
        """A config path as an absolute path under the repo root."""
        return REPO_ROOT / path
