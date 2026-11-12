"""The step-2 judge: pair the golden and the predicted records of one report, and grade each pair.

One call per report, with both lists as JSON (category left out: code grades it). The judge's pairs are checked in
code, as in step 1: one-to-one, valid ids, first pair wins; anything else is dropped with a reason. Code adds the
grades that need no judge: category, and whether the right mentions were merged (pages, first page).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict

from ..chat import Usage
from ..merge.mentions import Mention
from ..merge.schema import RiskRecord
from .config import JudgeConfig
from .golden import GoldenRecord
from .judge import Grade, yes_no

MERGE_JUDGE_FIELDS = ("title", "description", "mitigation")  # what the judge sees and grades


class MergePairVerdict(BaseModel):
    """The judge's verdict on one golden ↔ predicted pair."""

    model_config = ConfigDict(extra="forbid")

    golden_id: int
    pred_id: int
    title: Grade
    description: Grade
    mitigation: Grade
    reason: str


class MergeJudgement(BaseModel):
    """What the judge call returns."""

    model_config = ConfigDict(extra="forbid")

    pairs: list[MergePairVerdict]


def merge_payload(golden: list[GoldenRecord], records: list[RiskRecord]) -> str:
    """The judge's input: both lists, each record with an integer id, the judged fields and its pages."""

    def side(items: list[GoldenRecord] | list[RiskRecord]) -> list[dict]:
        return [
            {"id": i, **{f: getattr(r, f) for f in MERGE_JUDGE_FIELDS}, "pages": [loc.page for loc in r.locations]}
            for i, r in enumerate(items)
        ]

    return json.dumps({"golden": side(golden), "predicted": side(records)}, ensure_ascii=False, indent=1)


def merge_judge_chain(prompt: ChatPromptTemplate, model: BaseChatModel) -> Runnable:
    """(golden, records) -> {"raw": AIMessage, "parsed": MergeJudgement | None, "parsing_error": Exception | None}."""
    structured = model.with_structured_output(MergeJudgement, method="json_schema", strict=True, include_raw=True)
    return RunnableLambda(lambda case: {"payload": merge_payload(*case)}) | prompt | structured


class MergePair(BaseModel):
    """A valid golden ↔ predicted pair: the judge's grades and the code's."""

    golden_id: int
    pred_id: int
    title: Grade
    description: Grade
    mitigation: Grade
    category: Grade  # code: same primary category
    category_any: Grade  # code: predicted primary is the golden primary or secondary
    pages: Grade  # code: the record's pages are exactly the golden pages (the right mentions were merged)
    first_page: Grade  # code: the first location is the golden first one (the most informative mention)
    reason: str


class DroppedMergePair(BaseModel):
    verdict: MergePairVerdict
    why: str


def clean_merge_pairs(
    golden: list[GoldenRecord], records: list[RiskRecord], verdicts: list[MergePairVerdict]
) -> tuple[list[MergePair], list[DroppedMergePair]]:
    """Keep valid, one-to-one pairs (first wins) and add the code's grades; the rest is dropped with a reason."""
    pairs, dropped, used_g, used_p = [], [], set(), set()
    for v in verdicts:
        if not (0 <= v.golden_id < len(golden) and 0 <= v.pred_id < len(records)):
            dropped.append(DroppedMergePair(verdict=v, why="id out of range"))
        elif v.golden_id in used_g or v.pred_id in used_p:
            dropped.append(DroppedMergePair(verdict=v, why="risk already paired"))
        else:
            used_g.add(v.golden_id)
            used_p.add(v.pred_id)
            g, r = golden[v.golden_id], records[v.pred_id]
            g_pages, r_pages = [loc.page for loc in g.locations], [loc.page for loc in r.locations]
            pairs.append(
                MergePair(
                    **v.model_dump(),
                    category=yes_no(r.category == g.category),
                    category_any=yes_no(r.category in {g.category, g.secondary_category}),
                    pages=yes_no(set(r_pages) == set(g_pages)),
                    first_page=yes_no(r_pages[0] == g_pages[0]),
                )
            )
    return pairs, dropped


class MergeJudgeResult(BaseModel):
    """The judge's output for one merge run, checked."""

    pairs: list[MergePair] = []
    dropped: list[DroppedMergePair] = []
    usage: Usage = Usage()
    judge_error: str | None = None  # set: the run is not scored


def merge_judge_result(
    golden: list[GoldenRecord], records: list[RiskRecord], output: dict[str, Any] | Exception | None
) -> MergeJudgeResult:
    """The judge's output (None = not judged, because there are no records) as a MergeJudgeResult."""
    if output is None:
        return MergeJudgeResult()
    if isinstance(output, Exception):
        return MergeJudgeResult(judge_error=repr(output))
    usage = Usage.of(output["raw"])
    if output["parsed"] is None:  # a refusal, or a response that does not fit the schema
        return MergeJudgeResult(usage=usage, judge_error=f"no parsed output: {output['parsing_error']!r}")
    pairs, dropped = clean_merge_pairs(golden, records, output["parsed"].pairs)
    return MergeJudgeResult(pairs=pairs, dropped=dropped, usage=usage)


def judge_merge(
    chain: Runnable, golden: list[GoldenRecord], records: list[RiskRecord], config: JudgeConfig, eval_id: str
) -> MergeJudgeResult:
    """One judge call for the report; with no records nothing is sent, and a failed call is recorded, not raised."""
    if not records:
        return merge_judge_result(golden, records, None)
    run_config = {
        "run_name": "merge judge",
        "metadata": {"eval_id": eval_id},
    }
    try:
        output = chain.invoke((golden, records), config=run_config)
    except Exception as e:  # noqa: BLE001 - a failed judge call is recorded, not raised
        output = e
    return merge_judge_result(golden, records, output)


def miss_hint(g: GoldenRecord, records: list[RiskRecord], pairs: list[MergePair], mentions: list[Mention]) -> str:
    """Where a missed golden risk most likely got lost: in step 1, or merged into a record by step 2."""
    g_pages = {loc.page for loc in g.locations}
    if not g_pages & {m.page for m in mentions}:
        return "step 1: nothing extracted on its pages"
    sharing = [records[p.pred_id].id for p in pairs if g_pages & {loc.page for loc in records[p.pred_id].locations}]
    if sharing:
        return f"shares pages with {', '.join(sharing)}: merged into it, or missed by step 1 on that page"
    return "step 2: mentions on its pages, but no paired record"
