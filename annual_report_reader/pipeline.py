"""The pipeline, one function per step: what notebooks 01, 01b, 02 and 02b do cell by cell.

Each function takes its config and a chat model, makes the calls and returns the run or evaluation, unsaved. The
model is an argument so the wiring can be tested without the API; `cli.py` builds the Azure models and saves.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from .evaluation.config import JudgeConfig
from .evaluation.coverage import coverage
from .evaluation.golden import load_golden, load_golden_records
from .evaluation.judge import eval_id_for, judge_chain, judge_pages, load_judge_prompt, page_cases
from .evaluation.merge_checks import check_record
from .evaluation.merge_judge import judge_merge, merge_judge_chain
from .evaluation.runs import EvaluationRun, MergeEvaluationRun
from .evaluation.scores import Counts
from .extraction.chain import extract_groups, extraction_chain
from .extraction.config import ExtractionConfig
from .extraction.pages import adjacent_runs, render_pages
from .extraction.prompts import load_prompt
from .extraction.runs import ExtractionRun, run_id_for
from .extraction.schema import ExtractedRisks
from .merge.chain import merge_chain, merge_mentions
from .merge.config import MergeConfig
from .merge.mentions import mentions_from_run
from .merge.prompts import load_merge_prompt
from .merge.records import build_records
from .merge.runs import MergeRun, merge_run_id_for
from .merge.schema import MergedRisks

# The step-2 yardstick: JudgeConfig's defaults are step 1's.
MERGE_JUDGE = {"golden_path": Path("eval/golden/merging_and_summarizing.yaml"), "judge_prompt": "merge-judge"}


def extract(config: ExtractionConfig, model: BaseChatModel) -> ExtractionRun:
    """Step 1 (notebook 01): render the pages, one call per run of adjacent pages."""
    groups = adjacent_runs(render_pages(config.resolve(config.pdf_path), config.pages, config.dpi))
    definition = config.resolve(config.definition_path).read_text(encoding="utf-8")
    prompt = load_prompt(config.resolve(config.prompt_dir), config.prompt, definition)
    started_at = datetime.now(UTC)
    run_id = run_id_for(started_at)
    start = time.perf_counter()
    calls = extract_groups(extraction_chain(prompt, model), groups, config, run_id)
    return ExtractionRun(
        run_id=run_id,
        started_at=started_at,
        elapsed_s=round(time.perf_counter() - start, 1),
        config=config,
        system_prompt=prompt.format_messages(pages=[])[0].content,
        output_schema=ExtractedRisks.model_json_schema(),
        calls=calls,
    )


def evaluate_extraction(run: ExtractionRun, config: JudgeConfig, model: BaseChatModel) -> EvaluationRun:
    """Step-1 evaluation (notebook 01b): one judge call per page that is labelled, sent and has both sides."""
    golden = load_golden(config.resolve(config.golden_path))
    cov = coverage(run, golden)
    cases = page_cases(run, golden, cov.index[cov.scored].tolist())
    prompt = load_judge_prompt(config.resolve(config.judge_prompt_dir), config.judge_prompt)
    started_at = datetime.now(UTC)
    eval_id = eval_id_for(run.run_id, config.judge_prompt, started_at)
    start = time.perf_counter()
    pages = judge_pages(judge_chain(prompt, model), cases, config, eval_id)
    return EvaluationRun(
        eval_id=eval_id,
        started_at=started_at,
        elapsed_s=round(time.perf_counter() - start, 1),
        config=config,
        extraction_run_id=run.run_id,
        judge_system_prompt=prompt.format_messages(payload="")[0].content,
        coverage=cov.status.to_dict(),
        pages=pages,
    )


def merge(config: MergeConfig, run: ExtractionRun, model: BaseChatModel) -> MergeRun:
    """Step 2 (notebook 02): one call for the report. Refuses a run with failed pages, or another run than named."""
    if run.run_id != config.extraction_run:
        raise ValueError(f"config names {config.extraction_run}, got {run.run_id}")
    if run.failed_pages:
        raise ValueError(f"{run.run_id}: extraction failed on pages {run.failed_pages}; re-run step 1")
    mentions = mentions_from_run(run)
    prompt = load_merge_prompt(config.resolve(config.prompt_dir), config.prompt)
    started_at = datetime.now(UTC)
    run_id = merge_run_id_for(started_at)
    start = time.perf_counter()
    call = merge_mentions(merge_chain(prompt, model), mentions, config, run_id)
    records, problems = build_records(call.risks, mentions) if not call.error else ([], [])
    return MergeRun(
        run_id=run_id,
        started_at=started_at,
        elapsed_s=round(time.perf_counter() - start, 1),
        config=config,
        system_prompt=prompt.format_messages(mentions="")[0].content,
        output_schema=MergedRisks.model_json_schema(),
        mentions=mentions,
        call=call,
        problems=problems,
        records=records,
    )


def evaluate_merge(merge_run: MergeRun, config: JudgeConfig, model: BaseChatModel) -> MergeEvaluationRun:
    """Step-2 evaluation (notebook 02b): one judge call for the report, plus the code checks."""
    golden = load_golden_records(config.resolve(config.golden_path))
    prompt = load_judge_prompt(config.resolve(config.judge_prompt_dir), config.judge_prompt)
    started_at = datetime.now(UTC)
    eval_id = eval_id_for(merge_run.run_id, config.judge_prompt, started_at)
    start = time.perf_counter()
    result = judge_merge(merge_judge_chain(prompt, model), golden, merge_run.records, config, eval_id)
    return MergeEvaluationRun(
        eval_id=eval_id,
        started_at=started_at,
        elapsed_s=round(time.perf_counter() - start, 1),
        config=config,
        merge_run_id=merge_run.run_id,
        judge_system_prompt=prompt.format_messages(payload="")[0].content,
        result=result,
        checks=[check_record(r, merge_run.mentions) for r in merge_run.records],
        counts=Counts(golden=len(golden), predicted=len(merge_run.records), paired=len(result.pairs)),
    )
