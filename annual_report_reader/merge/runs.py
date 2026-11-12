"""A saved merge run (runs/merge/<run_id>.json): self-contained, so its evaluation needs nothing else."""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from .chain import MergeCallResult
from .config import MergeConfig
from .mentions import Mention
from .records import Problem
from .schema import RiskRecord

MERGE_RUNS_DIR = Path("runs/merge")  # relative to the repo root


def merge_run_id_for(started_at: datetime) -> str:
    """Sorts by start time; the random suffix keeps two runs started in the same second apart."""
    return f"{started_at:merge-%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"


class MergeRun(BaseModel):
    """One merge run: what was sent, what came back, and the final records."""

    run_id: str
    started_at: datetime
    elapsed_s: float
    config: MergeConfig
    system_prompt: str
    # The JSON schema sent as structured output: its field descriptions are instructions too.
    # None in runs saved before it was recorded.
    output_schema: dict | None = None
    mentions: list[Mention]
    call: MergeCallResult
    problems: list[Problem]
    records: list[RiskRecord]


def save_merge_run(merge_run: MergeRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{merge_run.run_id}.json"
    with path.open("x", encoding="utf-8") as f:  # a run is never overwritten
        f.write(merge_run.model_dump_json(indent=2) + "\n")
    return path


def load_merge_run(path: Path) -> MergeRun:
    return MergeRun.model_validate_json(path.read_text(encoding="utf-8"))
