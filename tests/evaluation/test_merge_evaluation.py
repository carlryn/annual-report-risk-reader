"""Deterministic parts of the step-2 evaluation (no API calls)."""

import json
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from annual_report_reader.evaluation.config import JudgeConfig
from annual_report_reader.evaluation.golden import GoldenRecord, load_golden_records
from annual_report_reader.evaluation.judge import eval_id_for
from annual_report_reader.evaluation.merge_checks import check_record, n_sentences
from annual_report_reader.evaluation.merge_judge import (
    MergeJudgement,
    MergePairVerdict,
    clean_merge_pairs,
    judge_merge,
    merge_judge_result,
    merge_payload,
    miss_hint,
)
from annual_report_reader.evaluation.runs import (
    EVAL_RUNS_DIR,
    MergeEvaluationRun,
    load_merge_eval,
    save_merge_eval,
)
from annual_report_reader.evaluation.scores import Counts, share_yes
from annual_report_reader.extraction.config import REPO_ROOT
from annual_report_reader.merge.mentions import Mention
from annual_report_reader.merge.schema import Location, RiskRecord

GOLDEN_YAML = """\
- title: Cyber security risks
  category: cyber
  secondary: operational
  locations:
    - { page: 118, section: "G1 Cyber security" }
    - { page: 51, section: "Main risks" }
  description: Attacks could disrupt operations.
  mitigation: Vestas invests in cyber resilience.
"""


def golden_record(title="Cyber security risks", category="cyber", secondary=None, pages=(118, 51)) -> GoldenRecord:
    return GoldenRecord(
        title=title,
        category=category,
        secondary_category=secondary,
        locations=[Location(page=p, section="s") for p in pages],
        description="d",
        mitigation="None",
    )


def record(
    rid="r00", title="Cyber", category="cyber", pages=(118, 51), mitigation="None", description="One."
) -> RiskRecord:
    return RiskRecord(
        id=rid,
        title=title,
        category=category,
        secondary_category=None,
        locations=[Location(page=p, section="s") for p in pages],
        description=description,
        mitigation=mitigation,
        mention_ids=[f"p{p}-0" for p in pages],
    )


def mention(page: int, mitigation: str = "None") -> Mention:
    return Mention(
        mention_id=f"p{page}-0",
        page=page,
        title="t",
        category="cyber",
        secondary_category=None,
        section="s",
        description="d",
        mitigation=mitigation,
    )


def verdict(g: int, p: int, **grades) -> MergePairVerdict:
    return MergePairVerdict(
        golden_id=g, pred_id=p, **({"title": "yes", "description": "yes", "mitigation": "yes"} | grades), reason="r"
    )


def raw() -> AIMessage:
    return AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


# --- golden ---------------------------------------------------------------------------------------------------


def test_load_golden_records_maps_secondary(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(GOLDEN_YAML, encoding="utf-8")
    (g,) = load_golden_records(path)
    assert g.secondary_category == "operational" and [loc.page for loc in g.locations] == [118, 51]


def test_golden_record_rejects_typos_and_missing_locations(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text(GOLDEN_YAML.replace("secondary:", "secondry:"), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_golden_records(path)
    with pytest.raises(ValidationError):
        golden_record(pages=())


def test_real_step2_golden_loads():
    assert len(load_golden_records(REPO_ROOT / "eval/golden/merging_and_summarizing.yaml")) == 11


# --- judge input ----------------------------------------------------------------------------------------------


def test_payload_has_ids_pages_and_no_category():
    data = json.loads(merge_payload([golden_record()], [record()]))
    assert data["golden"][0]["id"] == 0 and data["predicted"][0]["pages"] == [118, 51]
    assert "category" not in data["golden"][0] and "category" not in data["predicted"][0]


# --- pairs ----------------------------------------------------------------------------------------------------


def test_clean_pairs_one_to_one_and_ids_in_range():
    golden, records = [golden_record(), golden_record("Other")], [record(), record("r01")]
    pairs, dropped = clean_merge_pairs(golden, records, [verdict(0, 0), verdict(1, 0), verdict(5, 1)])
    assert [(p.golden_id, p.pred_id) for p in pairs] == [(0, 0)]
    assert [d.why for d in dropped] == ["risk already paired", "id out of range"]


def test_code_grades_category_and_pages():
    golden = [golden_record(category="cyber", secondary="operational", pages=(118, 51))]
    (same,), _ = clean_merge_pairs(golden, [record(category="operational", pages=(118, 51))], [verdict(0, 0)])
    assert (same.category, same.category_any, same.pages, same.first_page) == ("no", "yes", "yes", "yes")
    (moved,), _ = clean_merge_pairs(golden, [record(pages=(51, 118, 74))], [verdict(0, 0)])
    assert (moved.pages, moved.first_page) == ("no", "no")


def test_description_no_stays_a_pair():
    pairs, dropped = clean_merge_pairs([golden_record()], [record()], [verdict(0, 0, description="no")])
    assert len(pairs) == 1 and pairs[0].description == "no" and not dropped


def test_judge_result_shapes():
    assert merge_judge_result([], [], None).pairs == []
    assert merge_judge_result([], [], TimeoutError()).judge_error == "TimeoutError()"
    failed = merge_judge_result([], [], {"raw": raw(), "parsed": None, "parsing_error": ValueError("x")})
    assert "no parsed output" in failed.judge_error and failed.usage.input_tokens == 10
    ok = merge_judge_result(
        [golden_record()],
        [record()],
        {"raw": raw(), "parsed": MergeJudgement(pairs=[verdict(0, 0)]), "parsing_error": None},
    )
    assert ok.judge_error is None and len(ok.pairs) == 1


def test_judge_merge_sends_nothing_without_records():
    def must_not_run(_):
        raise AssertionError("judge called without records")

    result = judge_merge(RunnableLambda(must_not_run), [golden_record()], [], JudgeConfig(judge_deployment="sol"), "e")
    assert result.judge_error is None and result.pairs == []


def test_miss_hints():
    records = [record(pages=(51,))]
    pairs, _ = clean_merge_pairs([golden_record(pages=(51,))], records, [verdict(0, 0)])
    mentions = [mention(51), mention(73)]
    assert miss_hint(golden_record(pages=(85,)), records, pairs, mentions).startswith("step 1")
    assert miss_hint(golden_record(pages=(51,)), records, pairs, mentions).startswith("shares pages with r00")
    assert miss_hint(golden_record(pages=(73,)), records, pairs, mentions).startswith("step 2")


# --- checks ---------------------------------------------------------------------------------------------------


def test_sentences_ignore_abbreviations_and_decimals():
    assert n_sentences("Costs rose, e.g. steel. Margins fell to 6.39 percent. Risk remains.") == 3


def test_checks():
    mentions = [mention(118, mitigation="Vestas invests."), mention(51)]
    ok = check_record(record(mitigation="Vestas invests in resilience.", description="One. Two."), mentions)
    assert ok.passed and ok.sentences == 2
    dropped = check_record(record(mitigation="None"), mentions)  # a mention states one, the record does not
    assert not dropped.mitigation_consistent and not dropped.passed
    long = check_record(record(title="a b c d e f g h i j k", description="A. B. C. D.", pages=(51,)), [mention(51)])
    assert (long.description_max_3_sentences, long.title_max_10_words, long.mitigation_consistent) == (
        False,
        False,
        True,
    )


def test_share_yes():
    pairs, _ = clean_merge_pairs(
        [golden_record(), golden_record("B")], [record(), record("r01")], [verdict(0, 0), verdict(1, 1, title="no")]
    )
    assert share_yes(pairs, ["title", "description"]) == {"title": 0.5, "description": 1.0}
    assert share_yes([], ["title"]) == {"title": None}


# --- saved evaluation -----------------------------------------------------------------------------------------


def test_merge_eval_round_trip_and_never_overwrites(tmp_path):
    started = datetime(2026, 11, 12, 11, 0, tzinfo=UTC)
    golden, records = [golden_record()], [record()]
    evaluation = MergeEvaluationRun(
        eval_id=eval_id_for("merge-20261112T100000Z-ab12", "merge-judge", started),
        started_at=started,
        elapsed_s=3.0,
        config=JudgeConfig(judge_deployment="sol", judge_prompt="merge-judge"),
        merge_run_id="merge-20261112T100000Z-ab12",
        judge_system_prompt="j",
        result=merge_judge_result(
            golden, records, {"raw": raw(), "parsed": MergeJudgement(pairs=[verdict(0, 0)]), "parsing_error": None}
        ),
        checks=[check_record(records[0], [mention(118), mention(51)])],
        counts=Counts(golden=1, predicted=1, paired=1),
    )
    path = save_merge_eval(evaluation, tmp_path)
    assert load_merge_eval(path) == evaluation
    with pytest.raises(FileExistsError):
        save_merge_eval(evaluation, tmp_path)


@pytest.mark.parametrize("path", sorted((REPO_ROOT / EVAL_RUNS_DIR).glob("merge-*.json")), ids=lambda p: p.stem[:40])
def test_saved_merge_evaluations_load(path):
    assert load_merge_eval(path).counts.golden == 11
