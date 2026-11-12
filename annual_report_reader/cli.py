"""`arr-pipeline run`: extract → evaluate step 1 → merge → evaluate step 2, saving every run and evaluation.

    uv run arr-pipeline run                                   # the defaults: what notebooks 01-02b run
    uv run arr-pipeline run --model candidate_2 --effort default   # another extraction model, same judge
    uv run arr-pipeline run --extraction-run extract-...      # step 2 (and its evaluation) on a saved run
    uv run arr-pipeline run --no-eval                         # the pipeline alone, no judge calls
    uv run arr-pipeline repeat -n 5                           # 5 independent runs, then how much they differ
    uv run arr-pipeline repeat -n 5 --extraction-run extract-...   # the merge's stability on one extraction

A model is a deployment name, or an alias: `candidate_2` reads AZURE_OPENAI_DEPLOYMENT_CANDIDATE_2. Effort
`default` leaves it to the deployment (needed for non-reasoning models). The judges always come from JudgeConfig,
so runs with different extraction models are graded by the same yardstick.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .chat import ReasoningEffort, azure_chat_model
from .evaluation.config import JudgeConfig
from .evaluation.golden import load_golden_records
from .evaluation.runs import EVAL_RUNS_DIR, EvaluationRun, MergeEvaluationRun, save_eval, save_merge_eval
from .evaluation.scores import Counts
from .extraction.config import REPO_ROOT, ExtractionConfig
from .extraction.runs import RUNS_DIR, ExtractionRun, load_run, save_run
from .merge.config import MergeConfig
from .merge.runs import MERGE_RUNS_DIR, MergeRun, save_merge_run
from .pipeline import MERGE_JUDGE, evaluate_extraction, evaluate_merge, extract, merge
from .stability import attempt, format_report, report, save_report

EFFORTS = ("minimal", "low", "medium", "high", "default")


def deployment(name: str | None) -> str | None:
    """A deployment name, or an alias with an AZURE_OPENAI_DEPLOYMENT_<ALIAS> variable; None keeps the default."""
    if name is None:
        return None
    return os.getenv(f"AZURE_OPENAI_DEPLOYMENT_{name.upper()}", name)


def effort(value: str | None) -> dict[str, ReasoningEffort | None]:
    """The config override for an --effort flag: nothing, a level, or None for "default"."""
    if value is None:
        return {}
    return {"reasoning_effort": None if value == "default" else value}


def overrides(**fields) -> dict:
    """Only the flags that were given, so the configs keep their own defaults."""
    return {k: v for k, v in fields.items() if v is not None}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="arr-pipeline", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    add_pipeline_args(sub.add_parser("run", help="run the pipeline end to end"), no_eval=True)
    repeat = sub.add_parser("repeat", help="run the pipeline N times and report how the runs differ")
    repeat.add_argument("-n", "--runs", type=int, default=3, help="independent runs (default 3)")
    add_pipeline_args(repeat, no_eval=False)
    args = parser.parse_args(argv)
    if args.command == "repeat" and args.runs < 2:
        parser.error("repeat needs at least 2 runs")
    return args


def add_pipeline_args(run: argparse.ArgumentParser, no_eval: bool) -> None:
    step1 = run.add_argument_group("step 1: extraction")
    step1.add_argument("--extraction-run", help="skip step 1 and use this saved run (id in runs/extract/)")
    step1.add_argument("--pages", type=int, nargs="+", help="1-indexed PDF pages")
    step1.add_argument("--model", help="deployment or alias")
    step1.add_argument("--effort", choices=EFFORTS)
    step1.add_argument("--prompt", help="prompt name in prompts/")
    step2 = run.add_argument_group("step 2: merge")
    step2.add_argument("--merge-model", help="deployment or alias")
    step2.add_argument("--merge-effort", choices=EFFORTS)
    step2.add_argument("--merge-prompt", help="prompt name in prompts/")
    judge = run.add_argument_group("evaluation")
    judge.add_argument("--judge-model", help="deployment or alias, for both judges")
    if no_eval:
        judge.add_argument("--no-eval", action="store_true", help="no judge calls")


def saved(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def describe(c: Counts) -> str:
    if c.recall is None or c.precision is None:
        return f"{c.paired} of {c.golden} golden found, {c.predicted} predicted"
    return (
        f"recall {c.recall:.2f} | precision {c.precision:.2f} | F1 {c.f1:.2f} "
        f"({c.paired} of {c.golden} found, {c.predicted - c.paired} extra)"
    )


def report_extraction(run: ExtractionRun, path: Path | None) -> None:
    status = f"FAILED pages {run.failed_pages}" if run.failed_pages else "ok"
    where = f" → {saved(path)}" if path else " (saved earlier)"
    print(
        f"step 1  {run.run_id}: {len(run.risks)} risks, {run.config.deployment}, {status}, "
        f"{run.elapsed_s:.0f}s, ${run.config.cost(run.usage):.4f}{where}"
    )


def report_eval(ev: EvaluationRun, path: Path) -> None:
    failed = f" | judge failed on pages {ev.judge_failed_pages}" if ev.judge_failed_pages else ""
    print(f"eval 1  {describe(ev.total)}{failed} → {saved(path)}")


def report_merge(run: MergeRun, path: Path) -> None:
    status = f"FAILED: {run.call.error}" if run.call.error else f"{len(run.records)} records"
    problems = f", {len(run.problems)} problems" if run.problems else ""
    print(
        f"step 2  {run.run_id}: {len(run.mentions)} mentions → {status}{problems}, "
        f"{run.elapsed_s:.0f}s, ${run.config.cost(run.call.usage):.4f} → {saved(path)}"
    )


def report_merge_eval(ev: MergeEvaluationRun, path: Path) -> None:
    if ev.result.judge_error:
        print(f"eval 2  NOT SCORED: {ev.result.judge_error} → {saved(path)}")
        return
    checks = f"{sum(c.passed for c in ev.checks)}/{len(ev.checks)} records pass the checks"
    print(f"eval 2  {describe(ev.counts)} | {checks} → {saved(path)}")


@dataclass
class Outcome:
    """One pipeline run: every step's result (None where a step did not run)."""

    extraction: ExtractionRun
    extraction_eval: EvaluationRun | None
    merge: MergeRun | None
    merge_eval: MergeEvaluationRun | None

    @property
    def failed(self) -> bool:
        return bool(self.extraction.failed_pages) or self.merge is None or self.merge.call.error is not None


def run_once(args: argparse.Namespace) -> Outcome:
    """Every step once, each saved and reported as it finishes."""
    judge_fields = overrides(judge_deployment=deployment(args.judge_model))
    judge1 = JudgeConfig(**judge_fields)
    judge2 = JudgeConfig(**MERGE_JUDGE, **judge_fields)
    no_eval = getattr(args, "no_eval", False)
    judge_model = None if no_eval else azure_chat_model(judge1.judge_deployment, judge1.judge_effort)

    # Step 1
    ev1 = None
    if args.extraction_run:
        ext = load_run(REPO_ROOT / RUNS_DIR / f"{args.extraction_run}.json")
        report_extraction(ext, None)
    else:
        config = ExtractionConfig(
            **overrides(pages=args.pages, deployment=deployment(args.model), prompt=args.prompt),
            **effort(args.effort),
        )
        ext = extract(config, azure_chat_model(config.deployment, config.reasoning_effort))
        report_extraction(ext, save_run(ext, REPO_ROOT / RUNS_DIR))
        if judge_model:
            ev1 = evaluate_extraction(ext, judge1, judge_model)
            report_eval(ev1, save_eval(ev1, REPO_ROOT / EVAL_RUNS_DIR))
    if ext.failed_pages:
        print("stopped: step 2 needs an extraction run without failed pages", file=sys.stderr)
        return Outcome(ext, ev1, None, None)

    # Step 2
    merge_config = MergeConfig(
        extraction_run=ext.run_id,
        **overrides(deployment=deployment(args.merge_model), prompt=args.merge_prompt),
        **effort(args.merge_effort),
    )
    merged = merge(merge_config, ext, azure_chat_model(merge_config.deployment, merge_config.reasoning_effort))
    report_merge(merged, save_merge_run(merged, REPO_ROOT / MERGE_RUNS_DIR))
    ev2 = None
    if judge_model:
        ev2 = evaluate_merge(merged, judge2, judge_model)
        report_merge_eval(ev2, save_merge_eval(ev2, REPO_ROOT / EVAL_RUNS_DIR))
    return Outcome(ext, ev1, merged, ev2)


def repeat(args: argparse.Namespace, argv: list[str]) -> int:
    """N independent runs (nothing reused between them), then a comparison; the report is saved."""
    started_at = datetime.now(UTC)
    attempts, stopped, judge2 = [], 0, None
    for i in range(1, args.runs + 1):
        print(f"\n--- run {i} of {args.runs} ---")
        out = run_once(args)
        if out.failed or out.merge_eval is None:
            stopped += 1
            continue
        attempts.append(attempt(out.extraction, out.extraction_eval, out.merge, out.merge_eval))
        judge2 = out.merge_eval.config  # the same for every run
    if judge2 is None:
        print("no run finished; nothing to compare", file=sys.stderr)
        return 1
    golden = load_golden_records(judge2.resolve(judge2.golden_path))
    rep = report(started_at, argv, attempts, [g.title for g in golden])
    print(format_report(rep))
    if stopped:
        print(f"\n{stopped} of {args.runs} runs stopped early (a failed call) and are left out")
    print(f"\nSaved {saved(save_report(rep, REPO_ROOT / EVAL_RUNS_DIR))}")
    return 1 if stopped else 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    load_dotenv(REPO_ROOT / ".env")
    if args.command == "repeat":
        return repeat(args, argv)
    return 1 if run_once(args).failed else 0


if __name__ == "__main__":
    sys.exit(main())
