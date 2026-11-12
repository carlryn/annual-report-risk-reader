"""The step-1 judge: per page, pair golden and predicted risks and grade each pair.

One call per page with both lists as JSON (category and confidence left out, so they cannot bias matching). The
judge's pairs are checked in code: one-to-one, valid ids, first pair wins; anything else is dropped with a reason.
Category is graded in code. Pages with an empty side are decided in code, without a call.
"""

from __future__ import annotations

import json
import secrets
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict

from ..chat import Usage
from ..extraction.runs import ExtractionRun
from ..extraction.schema import Risk
from .config import JudgeConfig
from .golden import Golden, GoldenRisk

Grade = Literal["yes", "no"]
JUDGE_FIELDS = ("title", "section", "description", "mitigation")  # what the judge sees and grades


# --- what the judge returns -----------------------------------------------------------------------------------


class PairVerdict(BaseModel):
    """The judge's verdict on one golden ↔ predicted pair."""

    model_config = ConfigDict(extra="forbid")

    golden_id: int
    pred_id: int
    title: Grade
    section: Grade
    description: Grade
    mitigation: Grade
    reason: str


class PageJudgement(BaseModel):
    """What one judge call returns."""

    model_config = ConfigDict(extra="forbid")

    pairs: list[PairVerdict]


# --- what the judge gets --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PageCase:
    """One page to score: its golden and its predicted risks."""

    page: int
    golden: tuple[GoldenRisk, ...]
    predicted: tuple[Risk, ...]

    def payload(self) -> str:
        """The judge's input: both lists, each risk with an id and the judged fields only."""

        def side(risks):
            return [{"id": i, **{f: getattr(r, f) for f in JUDGE_FIELDS}} for i, r in enumerate(risks)]

        return json.dumps(
            {"page": self.page, "golden": side(self.golden), "predicted": side(self.predicted)},
            ensure_ascii=False,
            indent=1,
        )


def page_cases(run: ExtractionRun, golden: Golden, pages: list[int]) -> list[PageCase]:
    predicted = defaultdict(list)
    for r in run.risks:
        predicted[r.page].append(r)
    return [PageCase(p, tuple(golden[p]), tuple(predicted[p])) for p in pages]


def load_judge_prompt(prompt_dir: Path, name: str) -> ChatPromptTemplate:
    """The judge prompt `name`: the system file, then the page's JSON as the user message."""
    system = (prompt_dir / f"{name}.md").read_text(encoding="utf-8").removesuffix("\n")
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{payload}")])
    if prompt.input_variables != ["payload"]:  # e.g. a stray {placeholder} in the file
        raise ValueError(f"{name}: unexpected template variables {prompt.input_variables}")
    return prompt


def judge_chain(prompt: ChatPromptTemplate, model: BaseChatModel) -> Runnable:
    """PageCase -> {"raw": AIMessage, "parsed": PageJudgement | None, "parsing_error": Exception | None}."""
    structured = model.with_structured_output(PageJudgement, method="json_schema", strict=True, include_raw=True)
    return RunnableLambda(lambda case: {"payload": case.payload()}) | prompt | structured


# --- pairs and page results -----------------------------------------------------------------------------------


def yes_no(ok: bool) -> Grade:
    return "yes" if ok else "no"


class Pair(BaseModel):
    """A valid golden ↔ predicted pair: the judge's grades and the code's."""

    golden_id: int
    pred_id: int
    title: Grade
    section: Grade
    description: Grade
    mitigation: Grade
    category: Grade  # code: same primary category
    category_any: Grade  # code: predicted primary is the golden primary or secondary
    reason: str


class DroppedPair(BaseModel):
    verdict: PairVerdict
    why: str


def clean_pairs(case: PageCase, verdicts: list[PairVerdict]) -> tuple[list[Pair], list[DroppedPair]]:
    """Keep valid, one-to-one pairs (first wins) and add the category grades; the rest is dropped with a reason."""
    pairs, dropped, used_g, used_p = [], [], set(), set()
    for v in verdicts:
        if not (0 <= v.golden_id < len(case.golden) and 0 <= v.pred_id < len(case.predicted)):
            dropped.append(DroppedPair(verdict=v, why="id out of range"))
        elif v.golden_id in used_g or v.pred_id in used_p:
            dropped.append(DroppedPair(verdict=v, why="risk already paired"))
        else:
            used_g.add(v.golden_id)
            used_p.add(v.pred_id)
            g, p = case.golden[v.golden_id], case.predicted[v.pred_id]
            pairs.append(
                Pair(
                    **v.model_dump(),
                    category=yes_no(p.category == g.category),
                    category_any=yes_no(p.category in {g.category, g.secondary_category}),
                )
            )
    return pairs, dropped


class PageResult(BaseModel):
    """One scored page: the risks on both sides and how they were paired."""

    page: int
    golden: list[GoldenRisk]
    predicted: list[Risk]
    judged: bool  # False: one side was empty, decided in code
    pairs: list[Pair] = []
    dropped: list[DroppedPair] = []
    usage: Usage = Usage()
    judge_error: str | None = None  # set: the page is not scored


def page_result(case: PageCase, output: dict[str, Any] | Exception | None) -> PageResult:
    """The judge's output for `case` (None = not judged) as a PageResult."""
    base = {"page": case.page, "golden": list(case.golden), "predicted": list(case.predicted)}
    if output is None:
        return PageResult(**base, judged=False)
    if isinstance(output, Exception):
        return PageResult(**base, judged=True, judge_error=repr(output))
    usage = Usage.of(output["raw"])
    if output["parsed"] is None:  # a refusal, or a response that does not fit the schema
        return PageResult(
            **base, judged=True, usage=usage, judge_error=f"no parsed output: {output['parsing_error']!r}"
        )
    pairs, dropped = clean_pairs(case, output["parsed"].pairs)
    return PageResult(**base, judged=True, usage=usage, pairs=pairs, dropped=dropped)


# --- running --------------------------------------------------------------------------------------------------


def judge_pages(chain: Runnable, cases: list[PageCase], config: JudgeConfig, eval_id: str) -> list[PageResult]:
    """Score every case; only pages with both golden and predicted risks are sent to the judge."""
    to_judge = [c for c in cases if c.golden and c.predicted]
    run_configs = [
        {
            "run_name": f"judge p.{c.page}",
            "max_concurrency": config.max_concurrency,
            "metadata": {"eval_id": eval_id, "page": c.page},
        }
        for c in to_judge
    ]
    outputs = chain.batch(to_judge, config=run_configs, return_exceptions=True) if to_judge else []
    by_page = {c.page: out for c, out in zip(to_judge, outputs, strict=True)}
    return [page_result(c, by_page.get(c.page)) for c in cases]


def eval_id_for(run_id: str, judge_prompt: str, started_at: datetime) -> str:
    """Unique per evaluation: the judge is not deterministic, so re-scoring a run must not overwrite the last one."""
    return f"{run_id}--{judge_prompt}--{started_at:%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"
