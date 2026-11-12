"""How runs are graded: the golden set and the judge. The run being graded is not part of it (it is named where it
is evaluated), so the same config grades any run and also drives the judge's own tests.

The judge should not be the model it grades: it is a separate, configurable deployment, so a judge from another
provider can be dropped in later. Loading `.env` is the entry point's job.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from ..chat import ReasoningEffort, Usage
from ..extraction.config import REPO_ROOT


class JudgeConfig(BaseModel):
    """Everything that decides a grade. Frozen, and saved with each evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The yardstick
    golden_path: Path = Path("eval/golden/risk_extraction.yaml")

    # Judge
    judge_deployment: str = Field(default_factory=lambda: os.environ["AZURE_OPENAI_DEPLOYMENT_SOL"])
    judge_effort: ReasoningEffort | None = "medium"
    judge_prompt_dir: Path = Path("prompts")
    judge_prompt: str = "extract-judge"
    max_concurrency: PositiveInt = 4

    # Judge cost (USD per 1M tokens); None = unknown
    price_in_per_m: float | None = None
    price_out_per_m: float | None = None

    def cost(self, usage: Usage) -> float | None:
        """USD for `usage`, or None while the prices are unknown."""
        if self.price_in_per_m is None or self.price_out_per_m is None:
            return None
        return usage.cost(self.price_in_per_m, self.price_out_per_m)

    def resolve(self, path: Path) -> Path:
        """A config path as an absolute path under the repo root."""
        return REPO_ROOT / path
