"""Deterministic parts of annual_report_reader.extraction (no API calls)."""

from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from annual_report_reader.chat import Usage
from annual_report_reader.extraction.chain import CallResult, call_result
from annual_report_reader.extraction.config import REPO_ROOT, ExtractionConfig
from annual_report_reader.extraction.pages import PageGroup, PageImage, adjacent_runs
from annual_report_reader.extraction.prompts import load_prompt, pages_message
from annual_report_reader.extraction.runs import ExtractionRun, load_run, risks_per_page, run_id_for, save_run
from annual_report_reader.extraction.schema import ExtractedRisks, Risk


def config(**kw) -> ExtractionConfig:
    return ExtractionConfig(deployment="test-deployment", **kw)


def group(*pages: int) -> PageGroup:
    return PageGroup(tuple(PageImage(p, b"png") for p in pages))


def risk(page: int, title: str = "Cyber attack") -> Risk:
    return Risk(
        page=page,
        title=title,
        category="cyber",
        secondary_category=None,
        section="Main risks",
        description="An attack could stop production.",
        mitigation="None",
        confidence="high",
    )


# --- config ---------------------------------------------------------------------------------------------------


def test_config_sorts_pages():
    assert config(pages=(118, 50, 51)).pages == (50, 51, 118)


def test_config_rejects_duplicate_pages():
    with pytest.raises(ValidationError, match="duplicate pages: \\[51\\]"):
        config(pages=(51, 50, 51))


def test_config_is_frozen():
    with pytest.raises(ValidationError):
        config().dpi = 100


# --- pages ----------------------------------------------------------------------------------------------------


def test_adjacent_runs_groups_consecutive_pages_without_cap():
    images = [PageImage(p, b"") for p in (85, 71, 50, 72, 73, 74, 51, 118)]
    assert [g.pages for g in adjacent_runs(images)] == [[50, 51], [71, 72, 73, 74], [85], [118]]


def test_group_label():
    assert group(85).label == "p.85"
    assert group(71, 72, 73, 74).label == "p.71-74"


# --- prompts --------------------------------------------------------------------------------------------------


def test_prompt_fills_definition_and_only_needs_pages():
    c = config()
    prompt = load_prompt(c.resolve(c.prompt_dir), c.prompt, "MY DEFINITION")
    assert prompt.input_variables == ["pages"]
    system = prompt.format_messages(pages=[])[0].content
    assert "MY DEFINITION" in system and "{PRINCIPAL_RISK_DEFINITION}" not in system


def test_prompt_rejects_unknown_placeholder(tmp_path):
    (tmp_path / "v1.md").write_text("Hello {NAME}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected template variables"):
        load_prompt(tmp_path, "v1", "def")


def test_pages_message_labels_each_image():
    parts = pages_message(group(71, 72)).content
    assert [p["text"] for p in parts if p["type"] == "text"] == ["Page 71", "Page 72"]
    assert all(p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts if p["type"] == "image_url")


# --- chain results --------------------------------------------------------------------------------------------


def raw(input_tokens=100, output_tokens=50, reasoning=20) -> AIMessage:
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "output_token_details": {"reasoning": reasoning},
        },
    )


def test_call_result_ok():
    out = {"raw": raw(), "parsed": ExtractedRisks(risks=[risk(71)]), "parsing_error": None}
    res = call_result(group(71, 72), out)
    assert res.error is None and res.pages == [71, 72] and len(res.risks) == 1
    assert res.usage == Usage(input_tokens=100, output_tokens=50, reasoning_tokens=20)


def test_call_result_unparsed_keeps_usage():
    res = call_result(group(85), {"raw": raw(), "parsed": None, "parsing_error": ValueError("bad json")})
    assert res.risks == [] and "bad json" in res.error and res.usage.input_tokens == 100


def test_call_result_exception():
    res = call_result(group(118), TimeoutError("slow"))
    assert res.pages == [118] and "TimeoutError" in res.error and res.usage == Usage()


def test_usage_add_and_cost():
    total = Usage(input_tokens=1_000_000) + Usage(output_tokens=1_000_000)
    assert total.cost(0.2, 1.2) == pytest.approx(1.4)
    assert config(price_in_per_m=0.2, price_out_per_m=1.2).cost(total) == pytest.approx(1.4)


# --- runs -----------------------------------------------------------------------------------------------------


def make_run() -> ExtractionRun:
    started = datetime(2026, 11, 12, 9, 30, tzinfo=UTC)
    return ExtractionRun(
        run_id=run_id_for(started),
        started_at=started,
        elapsed_s=12.3,
        config=config(pages=(50, 51, 85)),
        system_prompt="system",
        calls=[
            CallResult(pages=[50, 51], risks=[risk(51), risk(51, "Tariffs")], usage=Usage(input_tokens=10)),
            CallResult(pages=[85], error="TimeoutError()"),
        ],
    )


def test_run_round_trip(tmp_path):
    run = make_run()
    path = save_run(run, tmp_path)
    assert path.name == f"{run.run_id}.json"
    assert load_run(path) == run


def test_run_ids_sort_by_time_and_differ_within_a_second():
    started = datetime(2026, 11, 12, 9, 30, tzinfo=UTC)
    a, b = run_id_for(started), run_id_for(started)
    assert a.startswith("extract-20261112T093000Z-") and a != b
    assert run_id_for(started) < run_id_for(started.replace(second=1))


def test_save_run_never_overwrites(tmp_path):
    run = make_run()
    save_run(run, tmp_path)
    with pytest.raises(FileExistsError):
        save_run(run, tmp_path)


def test_risks_per_page_lists_every_page_with_status():
    run = make_run()
    run.calls[0].risks.append(risk(52, "Outside"))  # the model named a page that was not sent
    df = risks_per_page(run)
    assert df.to_dict("index") == {
        50: {"risks": 0, "status": "ok"},
        51: {"risks": 2, "status": "ok"},
        52: {"risks": 1, "status": "not sent"},
        85: {"risks": 0, "status": "failed"},
    }


def test_repo_root_is_the_project():
    assert (REPO_ROOT / "pyproject.toml").exists()
