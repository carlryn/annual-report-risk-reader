"""A saved extraction run (runs/extract/<run_id>.json): the input of the merge and evaluation steps."""

from __future__ import annotations

import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from ..chat import Usage
from .chain import CallResult
from .config import ExtractionConfig
from .schema import Risk

RUNS_DIR = Path("runs/extract")  # relative to the repo root


def run_id_for(started_at: datetime) -> str:
    """Sorts by start time; the random suffix keeps two runs started in the same second apart."""
    return f"{started_at:extract-%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"


class ExtractionRun(BaseModel):
    """One extraction run: what was asked, and what came back."""

    run_id: str
    started_at: datetime
    elapsed_s: float
    config: ExtractionConfig
    system_prompt: str
    # The JSON schema sent as structured output: its field descriptions are instructions too.
    # None in runs saved before it was recorded.
    output_schema: dict | None = None
    calls: list[CallResult]

    @property
    def risks(self) -> list[Risk]:
        return [r for c in self.calls for r in c.risks]

    @property
    def usage(self) -> Usage:
        return sum((c.usage for c in self.calls), Usage())

    @property
    def failed_pages(self) -> list[int]:
        return [p for c in self.calls if c.error for p in c.pages]


def save_run(run: ExtractionRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run.run_id}.json"
    with path.open("x", encoding="utf-8") as f:  # a run is never overwritten
        f.write(run.model_dump_json(indent=2) + "\n")
    return path


def load_run(path: Path) -> ExtractionRun:
    return ExtractionRun.model_validate_json(path.read_text(encoding="utf-8"))


def risks_per_page(run: ExtractionRun) -> pd.DataFrame:
    """Risk count per page, with each page's status: ok, failed (its call failed) or not sent."""
    counts, failed = Counter(r.page for r in run.risks), set(run.failed_pages)
    rows = [
        {
            "page": p,
            "risks": counts.get(p, 0),
            "status": "failed" if p in failed else "ok" if p in run.config.pages else "not sent",
        }
        for p in sorted(set(run.config.pages) | set(counts))
    ]
    return pd.DataFrame(rows).set_index("page")
