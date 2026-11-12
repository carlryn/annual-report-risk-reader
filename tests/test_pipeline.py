"""The pipeline and the CLI end to end, with a fake chat model (no API calls)."""

import json
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from annual_report_reader import cli, pipeline
from annual_report_reader.evaluation.config import JudgeConfig
from annual_report_reader.evaluation.judge import PageJudgement, PairVerdict
from annual_report_reader.evaluation.merge_judge import MergeJudgement, MergePairVerdict
from annual_report_reader.extraction.chain import CallResult
from annual_report_reader.extraction.config import ExtractionConfig
from annual_report_reader.extraction.pages import PageImage
from annual_report_reader.extraction.runs import ExtractionRun
from annual_report_reader.extraction.schema import ExtractedRisks, Risk
from annual_report_reader.merge.config import MergeConfig
from annual_report_reader.merge.schema import MergedRisk, MergedRisks
from annual_report_reader.stability import StabilityReport, risk_stability

RISK = Risk(
    page=51,
    title="Geopolitical conflict",
    category="supply chain",
    secondary_category=None,
    section="Main risks",
    description="Conflicts disrupt supply chains.",
    mitigation="None",
    confidence="high",
)
GRADES = {"title": "yes", "description": "yes", "mitigation": "yes"}
RESPONSES: dict[type, Any] = {
    ExtractedRisks: ExtractedRisks(risks=[RISK]),
    PageJudgement: PageJudgement(pairs=[PairVerdict(golden_id=0, pred_id=0, section="yes", reason="r", **GRADES)]),
    MergedRisks: MergedRisks(
        risks=[
            MergedRisk(
                mention_ids=["p51-0"],
                title="Geopolitical conflict",
                category="supply chain",
                secondary_category=None,
                description="Conflicts disrupt supply chains.",
                mitigation="None",
            )
        ]
    ),
    MergeJudgement: MergeJudgement(pairs=[MergePairVerdict(golden_id=0, pred_id=0, reason="r", **GRADES)]),
}


class FakeModel(GenericFakeChatModel):
    """Structured output returns the canned answer for the schema, after formatting the real prompt."""

    calls: list = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs) -> RunnableLambda:
        def answer(prompt_value) -> dict:
            self.calls.append((schema.__name__, prompt_value.to_messages()))
            raw = AIMessage(
                content="{}", usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
            )
            return {"raw": raw, "parsed": RESPONSES[schema], "parsing_error": None}

        return RunnableLambda(answer)


def fake_model() -> FakeModel:
    return FakeModel(messages=iter([]))


@pytest.fixture
def no_pdf(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "render_pages", lambda path, pages, dpi: [PageImage(p, b"png") for p in pages])


def extraction_config(**kw) -> ExtractionConfig:
    return ExtractionConfig(deployment="luna", pages=(51,), **kw)


def judge(**kw) -> JudgeConfig:
    return JudgeConfig(judge_deployment="sol", **kw)


# --- steps ----------------------------------------------------------------------------------------------------


def test_extract_sends_the_definition_and_the_pages(no_pdf):
    model = fake_model()
    run = pipeline.extract(extraction_config(), model)
    assert [r.title for r in run.risks] == ["Geopolitical conflict"] and not run.failed_pages
    ((schema, messages),) = model.calls
    assert schema == "ExtractedRisks"
    assert "outside-in" in messages[0].content  # the principal-risk definition is filled in
    assert messages[1].content[0] == {"type": "text", "text": "Page 51"}


def test_evaluate_extraction(no_pdf):
    run = pipeline.extract(extraction_config(), fake_model())
    ev = pipeline.evaluate_extraction(run, judge(), fake_model())
    assert ev.extraction_run_id == run.run_id
    assert (ev.total.golden, ev.total.predicted, ev.total.paired) == (4, 1, 1)  # p.51 has 4 golden risks


def test_merge_and_evaluate_merge(no_pdf):
    run = pipeline.extract(extraction_config(), fake_model())
    model = fake_model()
    merged = pipeline.merge(MergeConfig(extraction_run=run.run_id, deployment="luna"), run, model)
    assert [r.mention_ids for r in merged.records] == [["p51-0"]] and not merged.problems
    assert json.loads(model.calls[0][1][1].content)[0]["mention_id"] == "p51-0"
    ev = pipeline.evaluate_merge(merged, judge(**pipeline.MERGE_JUDGE), fake_model())
    assert (ev.counts.golden, ev.counts.paired) == (11, 1) and ev.checks[0].passed


def test_merge_refuses_another_run_or_failed_pages():
    run = ExtractionRun(
        run_id="extract-a",
        started_at="2026-11-12T10:00:00Z",
        elapsed_s=1,
        config=extraction_config(),
        system_prompt="s",
        calls=[CallResult(pages=[51], error="TimeoutError()")],
    )
    with pytest.raises(ValueError, match="config names extract-b"):
        pipeline.merge(MergeConfig(extraction_run="extract-b", deployment="luna"), run, fake_model())
    with pytest.raises(ValueError, match="failed on pages \\[51\\]"):
        pipeline.merge(MergeConfig(extraction_run="extract-a", deployment="luna"), run, fake_model())


# --- cli ------------------------------------------------------------------------------------------------------


def test_deployment_aliases_and_effort(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_CANDIDATE_2", "gpt41mini")
    assert cli.deployment("candidate_2") == "gpt41mini"
    assert cli.deployment("my-deployment") == "my-deployment" and cli.deployment(None) is None
    assert cli.effort("default") == {"reasoning_effort": None} and cli.effort(None) == {}


@pytest.fixture
def offline_cli(monkeypatch, tmp_path, no_pdf) -> Callable[..., list[tuple[str, str, Any]]]:
    """The CLI with fake models, saving under tmp_path; returns the (deployment, effort) of every model built."""
    built = []

    def model(deployment, effort):
        built.append((deployment, effort))
        return fake_model()

    monkeypatch.setattr(cli, "azure_chat_model", model)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_LUNA", "luna")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_SOL", "sol")
    return built


def test_cli_runs_every_step_and_saves(offline_cli, tmp_path, capsys):
    assert cli.main(["run", "--pages", "51", "--effort", "low"]) == 0
    out = capsys.readouterr().out
    assert [line.split()[0] + " " + line.split()[1] for line in out.splitlines()] == [
        "step 1",
        "eval 1",
        "step 2",
        "eval 2",
    ]
    assert len(list((tmp_path / "runs/extract").glob("*.json"))) == 1
    assert len(list((tmp_path / "runs/merge").glob("*.json"))) == 1
    assert len(list((tmp_path / "eval/runs").glob("*.json"))) == 2
    assert offline_cli == [("sol", "medium"), ("luna", "low"), ("luna", "medium")]  # judge, extract, merge


def test_cli_no_eval_and_saved_extraction_run(offline_cli, tmp_path, capsys):
    assert cli.main(["run", "--pages", "51", "--no-eval"]) == 0
    (saved,) = (tmp_path / "runs/extract").glob("*.json")
    capsys.readouterr()
    assert cli.main(["run", "--extraction-run", saved.stem, "--no-eval"]) == 0
    assert "(saved earlier)" in capsys.readouterr().out
    assert not (tmp_path / "eval/runs").exists()
    assert len(list((tmp_path / "runs/merge").glob("*.json"))) == 2


def test_cli_repeat_compares_runs_and_saves_a_report(offline_cli, tmp_path, capsys):
    assert cli.main(["repeat", "-n", "2", "--pages", "51"]) == 0
    out = capsys.readouterr().out
    assert "=== Stability over 2 runs ===" in out
    assert "  2/2  p51 · Geopolitical conflict and trade barriers" in out  # found both times, same category
    assert " 0/2  p51 · Project execution" in out  # never found: listed, not left out
    (saved,) = (tmp_path / "eval/runs").glob("stability-*.json")
    rep = StabilityReport.model_validate_json(saved.read_text())
    assert len(rep.attempts) == 2 and rep.records == {1: 2} and rep.command[0] == "repeat"
    assert len(list((tmp_path / "runs/extract").glob("*.json"))) == 2  # independent runs, nothing reused


def test_cli_repeat_needs_two_runs():
    with pytest.raises(SystemExit):
        cli.parse_args(["repeat", "-n", "1"])


def test_risk_stability_marks_what_varies():
    found = [{"A": "cyber", "B": "market"}, {"A": "cyber", "B": "regulatory"}, {"A": "cyber"}]
    pages = [{"A": [118, 51]}, {"A": [118, 51]}, {"A": [51]}]
    a, b, c = risk_stability(found, ["A", "B", "C"], pages)
    assert (a.found, a.categories, a.pages) == (3, {"cyber": 3}, {"118, 51": 2, "51": 1}) and not a.stable
    assert (b.found, b.categories) == (2, {"market": 1, "regulatory": 1}) and not b.stable
    assert (c.found, c.categories) == (0, {}) and c.stable
