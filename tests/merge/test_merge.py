"""Deterministic parts of annual_report_reader.merge (no API calls)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from annual_report_reader.chat import Usage
from annual_report_reader.extraction.chain import CallResult
from annual_report_reader.extraction.config import REPO_ROOT, ExtractionConfig
from annual_report_reader.extraction.runs import ExtractionRun
from annual_report_reader.extraction.schema import Risk
from annual_report_reader.merge.chain import MergeCallResult, call_result
from annual_report_reader.merge.config import MergeConfig
from annual_report_reader.merge.mentions import Mention, mentions_from_run
from annual_report_reader.merge.prompts import load_merge_prompt, mentions_payload
from annual_report_reader.merge.records import build_records
from annual_report_reader.merge.runs import (
    MERGE_RUNS_DIR,
    MergeRun,
    load_merge_run,
    merge_run_id_for,
    save_merge_run,
)
from annual_report_reader.merge.schema import MergedRisk, MergedRisks


def config(**kw) -> MergeConfig:
    return MergeConfig(extraction_run="extract-20261112T093000Z", deployment="test-deployment", **kw)


def risk(page: int, title: str = "Cyber attack", mitigation: str = "None") -> Risk:
    return Risk(
        page=page,
        title=title,
        category="cyber",
        secondary_category=None,
        section=f"Section on p.{page}",
        description="An attack could stop production.",
        mitigation=mitigation,
        confidence="high",
    )


def extraction_run(*calls: CallResult) -> ExtractionRun:
    return ExtractionRun(
        run_id="extract-20261112T093000Z",
        started_at=datetime(2026, 11, 12, 9, 30, tzinfo=UTC),
        elapsed_s=1.0,
        config=ExtractionConfig(deployment="luna", pages=(51, 71, 118)),
        system_prompt="s",
        calls=list(calls),
    )


def mention(mention_id: str, page: int, title: str = "Cyber attack") -> Mention:
    return Mention(mention_id=mention_id, **risk(page, title).model_dump(exclude={"confidence"}))


def merged(*ids: str, title: str = "Cyber security") -> MergedRisk:
    return MergedRisk(
        mention_ids=list(ids),
        title=title,
        category="cyber",
        secondary_category=None,
        description="Merged.",
        mitigation="None",
    )


# --- config ---------------------------------------------------------------------------------------------------


def test_config_needs_an_extraction_run():
    with pytest.raises(ValidationError, match="extraction_run"):
        MergeConfig(deployment="test-deployment")


def test_config_is_frozen():
    with pytest.raises(ValidationError):
        config().prompt = "another-prompt"


# --- mentions -------------------------------------------------------------------------------------------------


def test_mention_ids_are_page_and_position_in_page_order():
    run = extraction_run(
        CallResult(pages=[118], risks=[risk(118, "A")]),
        CallResult(pages=[51], risks=[risk(51, "B"), risk(51, "C")]),
    )
    mentions = mentions_from_run(run)
    assert [(m.mention_id, m.title) for m in mentions] == [("p51-0", "B"), ("p51-1", "C"), ("p118-0", "A")]


def test_mentions_drop_confidence():
    (m,) = mentions_from_run(extraction_run(CallResult(pages=[51], risks=[risk(51)])))
    assert "confidence" not in m.model_dump()


# --- prompts --------------------------------------------------------------------------------------------------


def test_prompt_only_needs_mentions():
    c = config()
    prompt = load_merge_prompt(c.resolve(c.prompt_dir), c.prompt)
    assert prompt.input_variables == ["mentions"]


def test_prompt_rejects_unknown_placeholder(tmp_path):
    (tmp_path / "bad.md").write_text("Merge {these}.", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected template variables"):
        load_merge_prompt(tmp_path, "bad")


def test_payload_is_a_json_list_with_ids():
    data = json.loads(mentions_payload([mention("p51-0", 51)]))
    assert data[0]["mention_id"] == "p51-0" and data[0]["page"] == 51


# --- chain ----------------------------------------------------------------------------------------------------


def test_call_result_from_parsed_output():
    raw = AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    res = call_result({"raw": raw, "parsed": MergedRisks(risks=[merged("p51-0")]), "parsing_error": None})
    assert res.error is None and len(res.risks) == 1 and res.usage.input_tokens == 10


def test_call_result_from_failures():
    raw = AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    assert "no parsed output" in call_result({"raw": raw, "parsed": None, "parsing_error": ValueError("x")}).error
    assert call_result(TimeoutError()).error == "TimeoutError()"


# --- records --------------------------------------------------------------------------------------------------

MENTIONS = [mention("p51-0", 51), mention("p71-0", 71, "Carbon taxes"), mention("p118-0", 118)]


def test_records_copy_locations_in_the_models_order():
    records, problems = build_records([merged("p118-0", "p51-0"), merged("p71-0", title="Carbon")], MENTIONS)
    assert problems == []
    assert [r.id for r in records] == ["r00", "r01"]
    assert [loc.page for loc in records[0].locations] == [118, 51]
    assert records[0].locations[0].section == "Section on p.118"
    assert records[0].mention_ids == ["p118-0", "p51-0"]


def test_unknown_and_repeated_ids_are_dropped_first_use_wins():
    records, problems = build_records(
        [merged("p51-0", "p51-0", "p99-0"), merged("p51-0", "p118-0", title="Second"), merged("p71-0")], MENTIONS
    )
    assert [(p.kind, p.mention_id) for p in problems] == [
        ("repeated id", "p51-0"),
        ("unknown id", "p99-0"),
        ("repeated id", "p51-0"),
    ]
    assert [r.mention_ids for r in records] == [["p51-0"], ["p118-0"], ["p71-0"]]


def test_risk_without_valid_ids_is_dropped():
    records, problems = build_records([merged("p99-0", title="Ghost"), merged("p51-0", "p71-0", "p118-0")], MENTIONS)
    assert [(p.kind, p.risk_title) for p in problems] == [("unknown id", "Ghost"), ("no valid ids", "Ghost")]
    assert len(records) == 1


def test_left_out_mention_becomes_its_own_record():
    records, problems = build_records([merged("p51-0", "p118-0")], MENTIONS)
    assert [(p.kind, p.mention_id) for p in problems] == [("left out", "p71-0")]
    left = records[-1]
    assert (left.id, left.title, left.mention_ids) == ("r01", "Carbon taxes", ["p71-0"])
    assert left.description == MENTIONS[1].description


# --- runs -----------------------------------------------------------------------------------------------------


def make_run() -> MergeRun:
    started = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    records, problems = build_records([merged("p51-0", "p118-0")], MENTIONS)
    return MergeRun(
        run_id=merge_run_id_for(started),
        started_at=started,
        elapsed_s=2.0,
        config=config(),
        system_prompt="s",
        mentions=MENTIONS,
        call=MergeCallResult(risks=[merged("p51-0", "p118-0")], usage=Usage(input_tokens=10)),
        problems=problems,
        records=records,
    )


def test_run_ids_sort_by_time_and_differ_within_a_second():
    started = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
    a, b = merge_run_id_for(started), merge_run_id_for(started)
    assert a.startswith("merge-20261112T100000Z-") and a != b


def test_run_round_trip_and_never_overwrites(tmp_path):
    run = make_run()
    path = save_merge_run(run, tmp_path)
    assert load_merge_run(path) == run
    with pytest.raises(FileExistsError):
        save_merge_run(run, tmp_path)


@pytest.mark.parametrize("path", sorted((REPO_ROOT / MERGE_RUNS_DIR).glob("*.json")), ids=lambda p: Path(p).stem)
def test_saved_runs_load(path):
    assert load_merge_run(path).records
