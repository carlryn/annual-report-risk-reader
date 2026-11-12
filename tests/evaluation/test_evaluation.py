"""Deterministic parts of annual_report_reader.evaluation (no API calls)."""

import json
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from annual_report_reader.chat import Usage
from annual_report_reader.evaluation.config import JudgeConfig
from annual_report_reader.evaluation.coverage import coverage
from annual_report_reader.evaluation.golden import GoldenRisk, load_golden
from annual_report_reader.evaluation.judge import (
    PageCase,
    PageJudgement,
    PairVerdict,
    clean_pairs,
    eval_id_for,
    load_judge_prompt,
    page_cases,
    page_result,
)
from annual_report_reader.evaluation.runs import EvaluationRun, load_eval, save_eval
from annual_report_reader.evaluation.scores import Counts, scores_table, total
from annual_report_reader.extraction.chain import CallResult
from annual_report_reader.extraction.config import ExtractionConfig
from annual_report_reader.extraction.runs import ExtractionRun
from annual_report_reader.extraction.schema import Risk

GOLDEN_YAML = """
50: []
51:
  - title: Cyber attacks
    category: cyber
    section: Main risks
    description: An attack could stop production.
    mitigation: None
  - title: Tariffs
    category: regulatory
    secondary: market
    section: Main risks
    description: Tariffs raise costs.
    mitigation: Hedging.
"""


def gold(title="Cyber attacks", category="cyber", secondary_category=None, page=51) -> GoldenRisk:
    return GoldenRisk(
        page=page,
        title=title,
        category=category,
        secondary_category=secondary_category,
        section="Main risks",
        description="d",
        mitigation="None",
    )


def pred(title="Cyber attacks", category="cyber", page=51) -> Risk:
    return Risk(
        page=page,
        title=title,
        category=category,
        secondary_category=None,
        section="Main risks",
        description="d",
        mitigation="None",
        confidence="high",
    )


def verdict(g: int, p: int, **grades) -> PairVerdict:
    return PairVerdict(
        golden_id=g,
        pred_id=p,
        **({f: "yes" for f in ("title", "section", "description", "mitigation")} | grades),
        reason="r",
    )


def extraction_run(pages=(50, 51, 85), calls=None) -> ExtractionRun:
    return ExtractionRun(
        run_id="extract-20261112T093000Z",
        started_at=datetime(2026, 11, 12, 9, 30, tzinfo=UTC),
        elapsed_s=1.0,
        config=ExtractionConfig(deployment="luna", pages=pages),
        system_prompt="s",
        calls=calls or [CallResult(pages=[50, 51], risks=[pred(), pred("Extra")]), CallResult(pages=[85])],
    )


def judge_config() -> JudgeConfig:
    return JudgeConfig(judge_deployment="sol")


# --- golden ---------------------------------------------------------------------------------------------------


def test_load_golden_keeps_empty_pages(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(GOLDEN_YAML, encoding="utf-8")
    golden = load_golden(path)
    assert list(golden) == [50, 51] and golden[50] == []
    assert [r.title for r in golden[51]] == ["Cyber attacks", "Tariffs"]
    assert golden[51][1].secondary_category == "market" and golden[51][0].mitigation == "None"


def test_load_golden_rejects_unknown_keys_and_categories(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(GOLDEN_YAML.replace("secondary: market", "secondry: market"), encoding="utf-8")
    with pytest.raises(ValidationError, match="secondry"):
        load_golden(path)
    path.write_text(GOLDEN_YAML.replace("category: cyber", "category: cyberz"), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_golden(path)


def test_real_golden_file_loads():
    c = judge_config()
    golden = load_golden(c.resolve(c.golden_path))
    assert golden[50] == [] and golden[72] == []


# --- coverage -------------------------------------------------------------------------------------------------


def test_coverage_statuses():
    run = extraction_run(
        pages=(50, 51, 85, 90),
        calls=[CallResult(pages=[50, 51]), CallResult(pages=[85], error="Timeout"), CallResult(pages=[90])],
    )
    golden = {50: [], 51: [gold()], 85: [gold(page=85)], 118: [gold(page=118)]}
    cov = coverage(run, golden)
    assert cov.status.to_dict() == {50: "ok", 51: "ok", 85: "failed", 90: "not labelled", 118: "not sent"}
    assert cov.index[cov.scored].tolist() == [50, 51, 85]


# --- judge input ----------------------------------------------------------------------------------------------


def test_payload_hides_category_and_confidence():
    case = PageCase(51, (gold(),), (pred(),))
    data = json.loads(case.payload())
    assert data["page"] == 51
    for side in ("golden", "predicted"):
        assert set(data[side][0]) == {"id", "title", "section", "description", "mitigation"}


def test_page_cases_group_predictions_by_page():
    cases = page_cases(extraction_run(), {50: [], 51: [gold()], 85: [gold(page=85)]}, [50, 51, 85])
    assert [(c.page, len(c.golden), len(c.predicted)) for c in cases] == [(50, 0, 0), (51, 1, 2), (85, 1, 0)]


def test_judge_prompt_takes_only_payload(tmp_path):
    c = judge_config()
    prompt = load_judge_prompt(c.resolve(c.judge_prompt_dir), c.judge_prompt)
    assert prompt.input_variables == ["payload"]
    (tmp_path / "j.md").write_text("Judge {THING}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected template variables"):
        load_judge_prompt(tmp_path, "j")


# --- pairs ----------------------------------------------------------------------------------------------------


def test_clean_pairs_drops_invalid_and_duplicates():
    case = PageCase(51, (gold(), gold("Tariffs")), (pred(), pred("Tariffs")))
    pairs, dropped = clean_pairs(case, [verdict(0, 0), verdict(5, 1), verdict(1, 0), verdict(1, 1)])
    assert [(p.golden_id, p.pred_id) for p in pairs] == [(0, 0), (1, 1)]
    assert [d.why for d in dropped] == ["id out of range", "risk already paired"]


def test_clean_pairs_grades_category_in_code():
    case = PageCase(51, (gold("Tariffs", "regulatory", secondary_category="market"),), (pred("Tariffs", "market"),))
    (pair,), _ = clean_pairs(case, [verdict(0, 0)])
    assert (pair.category, pair.category_any) == ("no", "yes")


def raw() -> AIMessage:
    return AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def test_page_result_paths():
    case = PageCase(51, (gold(),), (pred(),))
    assert page_result(case, None).judged is False
    assert "Timeout" in page_result(case, TimeoutError()).judge_error
    unparsed = page_result(case, {"raw": raw(), "parsed": None, "parsing_error": ValueError("x")})
    assert unparsed.judge_error and unparsed.usage.input_tokens == 10
    ok = page_result(case, {"raw": raw(), "parsed": PageJudgement(pairs=[verdict(0, 0)]), "parsing_error": None})
    assert ok.judge_error is None and len(ok.pairs) == 1 and ok.usage == Usage(input_tokens=10, output_tokens=5)


# --- scores ---------------------------------------------------------------------------------------------------


def test_counts_scores():
    assert Counts(golden=4, predicted=2, paired=2).model_dump() == {
        "golden": 4,
        "predicted": 2,
        "paired": 2,
        "recall": 0.5,
        "precision": 1.0,
        "f1": pytest.approx(2 / 3),
    }
    empty = Counts()
    assert (empty.recall, empty.precision, empty.f1) == (None, None, None)
    assert Counts(golden=1, predicted=1, paired=0).f1 == 0.0


def test_total_is_micro_and_skips_failed_judge_pages():
    results = [
        page_result(PageCase(50, (), (pred(page=50),)), None),  # an extra on a no-risk page
        page_result(
            PageCase(51, (gold(),), (pred(),)),
            {"raw": raw(), "parsed": PageJudgement(pairs=[verdict(0, 0)]), "parsing_error": None},
        ),
        page_result(PageCase(85, (gold(page=85),), (pred(page=85),)), TimeoutError()),
    ]
    assert total(results) == Counts(golden=1, predicted=2, paired=1)
    table = scores_table(results)
    assert table.index.tolist() == ["p.50", "p.51", "all"] and table.golden.dtype == "int64"


# --- saved evaluation -----------------------------------------------------------------------------------------


def test_eval_round_trip(tmp_path):
    started = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    case = PageCase(51, (gold(),), (pred(),))
    evaluation = EvaluationRun(
        eval_id=eval_id_for("extract-20261112T093000Z", "extract-judge", started),
        started_at=started,
        elapsed_s=2.0,
        config=judge_config(),
        extraction_run_id="extract-20261112T093000Z",
        judge_system_prompt="j",
        coverage={51: "ok"},
        pages=[
            page_result(case, {"raw": raw(), "parsed": PageJudgement(pairs=[verdict(0, 0)]), "parsing_error": None})
        ],
    )
    path = save_eval(evaluation, tmp_path)
    assert path.name.startswith("extract-20261112T093000Z--extract-judge--20261112T100000Z-")
    assert json.loads(path.read_text())["total"]["recall"] == 1.0
    assert load_eval(path) == evaluation
    with pytest.raises(FileExistsError):
        save_eval(evaluation, tmp_path)
