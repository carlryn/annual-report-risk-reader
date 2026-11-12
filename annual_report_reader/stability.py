"""Run-to-run stability: the same pipeline run several times on the same input, compared.

The model is not deterministic (the deployment ignores temperature), so consistency is measured, not assumed. Each
repetition is a full, independent run (nothing is reused); this module only compares their evaluations: which golden
risks each one found, with which category, and (step 2) which pages were merged into each record.
"""

from __future__ import annotations

import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from .evaluation.golden import load_golden_records
from .evaluation.runs import EvaluationRun, MergeEvaluationRun
from .evaluation.scores import Counts
from .extraction.runs import ExtractionRun
from .merge.runs import MergeRun


class Attempt(BaseModel):
    """One repetition, reduced to what is compared across repetitions."""

    extraction_run_id: str
    merge_run_id: str
    step1: Counts | None  # None: step 1 was not re-run (a saved extraction run was reused)
    step2: Counts | None  # None: the judge failed
    step1_golden: list[str]  # "p51 · <golden title>" for every golden risk on a scored page
    step1_found: dict[str, str]  # "p51 · <golden title>" -> predicted primary category
    step2_found: dict[str, str]  # golden title -> predicted primary category
    step2_pages: dict[str, list[int]]  # golden title -> pages of the paired record
    records: int
    problems: int
    cost: float  # extraction (when re-run) + merge, USD; judges not included


def attempt(ext: ExtractionRun, ev1: EvaluationRun | None, merged: MergeRun, ev2: MergeEvaluationRun) -> Attempt:
    step1_golden, step1_found = [], {}
    if ev1 is not None:
        for page in ev1.pages:
            step1_golden += [f"p{page.page} · {g.title}" for g in page.golden]
            for p in page.pairs:
                step1_found[f"p{page.page} · {page.golden[p.golden_id].title}"] = page.predicted[p.pred_id].category
    golden = load_golden_records(ev2.config.resolve(ev2.config.golden_path))
    pairs = [] if ev2.result.judge_error else ev2.result.pairs
    return Attempt(
        extraction_run_id=ext.run_id,
        merge_run_id=merged.run_id,
        step1=None if ev1 is None else ev1.total,
        step2=None if ev2.result.judge_error else ev2.counts,
        step1_golden=step1_golden,
        step1_found=step1_found,
        step2_found={golden[p.golden_id].title: merged.records[p.pred_id].category for p in pairs},
        step2_pages={
            golden[p.golden_id].title: [loc.page for loc in merged.records[p.pred_id].locations] for p in pairs
        },
        records=len(merged.records),
        problems=len(merged.problems),
        cost=round((0.0 if ev1 is None else ext.config.cost(ext.usage)) + merged.config.cost(merged.call.usage), 4),
    )


class RiskStability(BaseModel):
    """One golden risk across the repetitions."""

    risk: str
    found: int  # in how many repetitions it was found
    categories: dict[str, int]  # predicted category -> repetitions
    pages: dict[str, int] = {}  # step 2: merged pages ("118, 51, 74") -> repetitions

    @property
    def stable(self) -> bool:
        return len(self.categories) <= 1 and len(self.pages) <= 1


def risk_stability(
    found: list[dict[str, str]], golden: list[str], pages: list[dict] | None = None
) -> list[RiskStability]:
    out = []
    for risk in golden:
        hits = [f[risk] for f in found if risk in f]
        page_sets = [", ".join(map(str, p[risk])) for p in pages or [] if risk in p]
        out.append(
            RiskStability(risk=risk, found=len(hits), categories=dict(Counter(hits)), pages=dict(Counter(page_sets)))
        )
    return out


class Spread(BaseModel):
    min: float
    mean: float
    max: float


def spread(values: list[float]) -> Spread | None:
    return Spread(min=min(values), mean=round(mean(values), 3), max=max(values)) if values else None


class StabilityReport(BaseModel):
    """N repetitions of the pipeline, compared. Saved to eval/runs/stability-<time>-<suffix>.json."""

    report_id: str
    started_at: datetime
    command: list[str]
    attempts: list[Attempt]
    step1_risks: list[RiskStability]
    step2_risks: list[RiskStability]
    step1_f1: Spread | None
    step2_f1: Spread | None
    records: dict[int, int]  # number of records -> repetitions


def report(
    started_at: datetime, command: list[str], attempts: list[Attempt], golden_step2: list[str]
) -> StabilityReport:
    step1_keys = list(dict.fromkeys(k for a in attempts for k in a.step1_golden))  # page order, as evaluated
    return StabilityReport(
        report_id=f"{started_at:stability-%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}",
        started_at=started_at,
        command=command,
        attempts=attempts,
        step1_risks=risk_stability([a.step1_found for a in attempts if a.step1], step1_keys),
        step2_risks=risk_stability(
            [a.step2_found for a in attempts if a.step2], golden_step2, [a.step2_pages for a in attempts if a.step2]
        ),
        step1_f1=spread([a.step1.f1 for a in attempts if a.step1 and a.step1.f1 is not None]),
        step2_f1=spread([a.step2.f1 for a in attempts if a.step2 and a.step2.f1 is not None]),
        records=dict(sorted(Counter(a.records for a in attempts).items())),
    )


def save_report(rep: StabilityReport, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{rep.report_id}.json"
    with path.open("x", encoding="utf-8") as f:
        f.write(rep.model_dump_json(indent=2) + "\n")
    return path


def format_report(rep: StabilityReport) -> str:
    """The report as plain text: one line per repetition, then every golden risk across them."""
    n = len(rep.attempts)
    lines = [f"\n=== Stability over {n} runs ===", ""]
    for i, a in enumerate(rep.attempts, 1):
        s1 = (
            f"step 1 F1 {a.step1.f1:.2f} ({a.step1.paired}/{a.step1.golden}, {a.step1.predicted - a.step1.paired} extra)"
            if a.step1 and a.step1.f1 is not None
            else "step 1 reused"
        )
        s2 = (
            f"F1 {a.step2.f1:.2f} ({a.step2.paired}/{a.step2.golden}, {a.step2.predicted - a.step2.paired} extra)"
            if a.step2 and a.step2.f1 is not None
            else "NOT SCORED"
        )
        lines.append(f"run {i}: {s1} | step 2 {s2} | {a.records} records, {a.problems} problems | ${a.cost:.4f}")
    if rep.step1_f1:
        lines.append(
            f"\nstep 1 F1: min {rep.step1_f1.min:.2f}, mean {rep.step1_f1.mean:.2f}, max {rep.step1_f1.max:.2f}"
        )
    if rep.step2_f1:
        lines.append(f"step 2 F1: min {rep.step2_f1.min:.2f}, mean {rep.step2_f1.mean:.2f}, max {rep.step2_f1.max:.2f}")
    lines.append(f"records per run: {', '.join(f'{k} ({v}x)' for k, v in rep.records.items())}")

    def table(title: str, risks: list[RiskStability], runs: int) -> None:
        if not runs:
            return
        lines.extend(["", f"{title} (found in k of {runs} runs; * = varies between runs)"])
        for r in risks:
            mark = " " if r.found in (0, runs) and r.stable else "*"
            cats = ", ".join(f"{c} {k}x" if len(r.categories) > 1 else c for c, k in r.categories.items()) or "-"
            pages = ", ".join(f"p.{p} {k}x" if len(r.pages) > 1 else f"p.{p}" for p, k in r.pages.items())
            lines.append(f" {mark} {r.found}/{runs}  {r.risk:<62} {cats:<28} {pages}")

    table("Step 1, per page", rep.step1_risks, sum(1 for a in rep.attempts if a.step1))
    table("Step 2, final records", rep.step2_risks, sum(1 for a in rep.attempts if a.step2))
    return "\n".join(lines)
