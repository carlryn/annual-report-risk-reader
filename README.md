# annual-report-reader

Principal-risk extraction from annual reports.

- [PLAN.md](PLAN.md): the product thinking (user, surface, what we optimise for, assumptions, what's deferred)
- [DESIGN.md](DESIGN.md): stack, parsing, decomposition, evaluation, cost and scaling
- [STRETCH.md](STRETCH.md): what I'd build next

Code is written in Jupyter notebooks first, then lifted into a small package with tests. Two steps, each with its
own notebook and evaluation notebook: extraction (01, 01b) and merge (02, 02b).

```text
notebooks/
  00_parsing_comparison.ipynb      how to read a page: text layer vs font-aware text vs OCR vs page image → risks
  01_risk_extraction.ipynb         step 1: principal risks from page images → runs/extract/<run_id>.json
  01b_extraction_eval.ipynb        step 1 scored against the golden set by an LLM judge → eval/runs/
  02_risk_merge.ipynb              step 2: mentions of one extraction run → unique risk records → runs/merge/
  02b_merge_eval.ipynb             step 2 scored against the step-2 golden set + code checks → eval/runs/
  03_page_locator.ipynb            screens all 195 pages: which could hold principal risks → eval/runs/
prompts/                           one prompt file per step, shared by notebooks and package (each run stores the prompt it sent)
annual_report_reader/
  chat.py                          LangChain Azure OpenAI chat model, token usage
  pipeline.py, cli.py              one function per step; `arr-pipeline run` chains them and saves every run
  extraction/                      step 1: config → schema → pages → prompts → chain → runs
  merge/                           step 2: config → mentions → schema → prompts → chain → records → runs
  evaluation/                      both steps' evaluation: judges, golden sets, code checks, scores, saved runs
eval/golden/                       golden sets: risk_extraction.yaml (step 1, per page), merging_and_summarizing.yaml (step 2)
eval/runs/                         saved evaluations and page-locator runs
runs/extract/, runs/merge/         saved extraction and merge runs
configs/reports/vestas-2025.json   report metadata (company, sector, sections); not read by the code yet
tests/                             pytest for the package (no LLM calls)
```

## Run

```bash
cp .env.example .env     # fill in the Azure OpenAI endpoint, key and deployments
uv sync
uv run pytest            # no API calls
uv run arr-pipeline run  # extract → evaluate step 1 → merge → evaluate step 2, every run saved
```

```text
step 1  extract-…: 13 risks, gpt-5.6-luna-carl-test, ok, 18s, $0.0154 → runs/extract/extract-….json
eval 1  recall 0.93 | precision 1.00 | F1 0.96 (13 of 14 found, 0 extra) → eval/runs/extract-…--extract-judge--….json
step 2  merge-…: 13 mentions → 9 records, 8s, $0.0022 → runs/merge/merge-….json
eval 2  recall 0.82 | precision 1.00 | F1 0.90 (9 of 11 found, 0 extra) | 9/9 records pass the checks → eval/runs/…
```

Options (`uv run arr-pipeline run --help`): `--model`, `--effort`, `--prompt`, `--pages` for step 1;
`--merge-model`, `--merge-effort`, `--merge-prompt` for step 2; `--judge-model`, `--no-eval`; and
`--extraction-run <id>` to run step 2 on a saved extraction run.

`uv run arr-pipeline repeat -n 5 [same options]` runs the pipeline 5 times, independently, and reports how the runs
differ: F1 per run and its spread, and for every golden risk in how many runs it was found, with which category and
which merged pages (`*` marks what varies). The report is saved to `eval/runs/stability-….json`. A model is a deployment name or an alias
(`mini` reads `AZURE_OPENAI_DEPLOYMENT_MINI`); `--effort default` leaves it to the deployment.

### Compare models: 10 runs each

```bash
# Luna (the default for extraction and merge)
uv run arr-pipeline repeat -n 10

# GPT-4.1 mini for extraction and merge
uv run arr-pipeline repeat -n 10 --model mini --effort default --merge-model mini --merge-effort default
```

The judge is Sol in both, so the scores are comparable. GPT-4.1 mini is not a reasoning model, so it needs
`--effort default` (and `--merge-effort default`): the defaults, `high` and `medium`, send a reasoning effort it
rejects. To swap only the extraction model, drop the two `--merge-…` options and step 2 stays on Luna. Each command
writes its own `eval/runs/stability-….json`.

The notebooks show the same steps stage by stage: open them in VS Code with the .venv kernel and run 01 → 01b,
then 02 → 02b. The CLI and the notebooks call the same step modules; `annual_report_reader/pipeline.py` strings them
together one function per step, as the notebooks do cell by cell.

Settings live in `ExtractionConfig`, `MergeConfig` and `JudgeConfig` (pages, DPI, deployment, reasoning effort,
prompt, prices). Each step names the run it reads explicitly. Each saved run holds its full config and
system prompt, so a result can be traced to exactly what produced it. LLM responses are never cached: every run calls the API.

## Output

Step 2 writes one record per unique principal risk (`records` in `runs/merge/<run_id>.json`; schema in
`annual_report_reader/merge/schema.py`). One record from a saved run, mitigation shortened:

```json
{
  "id": "r02",
  "title": "Cyber security risks",
  "category": "cyber",
  "secondary_category": "operational",
  "locations": [
    {"page": 118, "section": "Cyber security – SBM-3 Impacts, risks, and opportunities"},
    {"page": 51, "section": "Main risks - Cyber attacks"},
    {"page": 74, "section": "G1 Business Conduct – Financial risks or opportunities"}
  ],
  "description": "Vestas’ digitally connected wind turbines, critical assets, infrastructure, data, technology, and operations may be targeted by cybercriminals or geopolitical actors. Successful attacks could disrupt Vestas’ operations and customers’ energy production, cause outages or physical damage, expose intellectual property, create legal liability, erode partner trust, and result in significant financial and reputational losses.",
  "mitigation": "Vestas has increased cybersecurity investment across R&D, service, and digital infrastructure and applies an enterprise-wide Cyber Risk Management Strategy covering strategic, customer, manufacturing, and operational domains. …",
  "mention_ids": ["p118-0", "p51-2", "p74-1"]
}
```

The case asks for "the section and page where you found it". A merged record is found in several places, so
`locations` is a list rather than one section and page: one entry per step-1 mention merged into the record, most
informative first. The pages and sections are copied in code from those mentions, not written by the model.
`mention_ids` points back to the step-1 mentions.
