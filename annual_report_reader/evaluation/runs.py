"""Saved evaluations (eval/runs/<run>--<judge prompt>--<time>.json): of an extraction run (step 1) or of a merge
run (step 2). Every evaluation is kept."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, computed_field

from .config import JudgeConfig
from .judge import PageResult
from .merge_checks import RecordChecks
from .merge_judge import MergeJudgeResult
from .scores import Counts, total

EVAL_RUNS_DIR = Path("eval/runs")  # relative to the repo root


class EvaluationRun(BaseModel):
    """One evaluation of one extraction run."""

    eval_id: str
    started_at: datetime
    elapsed_s: float
    config: JudgeConfig
    extraction_run_id: str
    judge_system_prompt: str
    coverage: dict[int, str]  # page -> ok | failed | not sent | not labelled
    pages: list[PageResult]

    @computed_field
    @property
    def total(self) -> Counts:
        return total(self.pages)

    @property
    def judge_failed_pages(self) -> list[int]:
        return [r.page for r in self.pages if r.judge_error]


def save_eval(evaluation: EvaluationRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{evaluation.eval_id}.json"
    with path.open("x", encoding="utf-8") as f:  # an evaluation is never overwritten
        f.write(evaluation.model_dump_json(indent=2) + "\n")
    return path


def load_eval(path: Path) -> EvaluationRun:
    return EvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))


class MergeEvaluationRun(BaseModel):
    """One evaluation of one merge run."""

    eval_id: str
    started_at: datetime
    elapsed_s: float
    config: JudgeConfig
    merge_run_id: str
    judge_system_prompt: str
    result: MergeJudgeResult
    checks: list[RecordChecks]
    counts: Counts  # golden = golden records, predicted = records, paired = valid pairs


def save_merge_eval(evaluation: MergeEvaluationRun, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{evaluation.eval_id}.json"
    with path.open("x", encoding="utf-8") as f:  # an evaluation is never overwritten
        f.write(evaluation.model_dump_json(indent=2) + "\n")
    return path


def load_merge_eval(path: Path) -> MergeEvaluationRun:
    return MergeEvaluationRun.model_validate_json(path.read_text(encoding="utf-8"))
