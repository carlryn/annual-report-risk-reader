# DESIGN

I focused on three things: parsing, principal-risk extraction, and evaluation. Every choice below traces back to PLAN: **recall first**, **consistency and grounding**, and **cheap to run**.

## Stack

- Python 3.12 with `uv`. PyMuPDF renders the PDF pages.
- LangChain (`langchain-openai`) on Azure OpenAI, with strict structured output (Pydantic schemas), so every response is valid JSON in a fixed shape. The extraction model is GPT-5.6-Luna. The judge is GPT-5.6-Sol. Although they are two different models, they have likely seen the same training data. This is not the best approach; the judge should come from another provider. The judge is its own configured chat model (`JudgeConfig`), so swapping it for another provider is a config change.
- Code is written in notebooks first and then lifted into the `annual_report_reader` package (`extraction/`, `merge/`, `evaluation/`), with tests. `arr-pipeline run` runs both steps and both evaluations in one command. Every run is saved with its full config and prompt, and LLM responses are never cached or reused.

## How I worked

1. Figuring out the best option for parsing the PDFs.
2. Deciding on how to extract information.
3. Building the golden datasets.
4. Building the LLM workflow including judges.

## Parsing: page images, not the text layer

When an LLM gets something wrong, the cause is often bad context. Parsing PDFs is complex and various approaches exist. What works for one document can fail for another. I believe proper parsing is absolutely critical for this task.

I evaluated four approaches. You can read the notebook `notebooks/00_parsing_comparison.ipynb` to see the results.

- Parsing the text layer directly (plain, and with font sizes turned into headings). It does not read the text in correct reading order.
- OCR with Luna as a vision model. Works better, but I noticed some informative features went missing in the context. Information loss which will create mistakes at some point.
- Directly extracting the risks with Luna receiving images just as in OCR. The choice I went with.
- An alternative I did not consider for this case is Document Layout Extraction. This works over both the image and the text layer but is a more complex technique. I did not think it was worth it given the time constraint.

Each page is rendered to an image (200 DPI) and sent to Luna, which reads the layout the way a person does.
Images cost more input tokens than text (about 2.4x on p.71), but LLM costs are not major. On p.71 alone, with today's prompt, the text layer finds the same single risk as the image; the case for images rests on the layout-heavy pages (p.51's three columns, the p.71-74 table).

## Decomposition: two steps

**Step 1: extract, multiple pages that come in order.** Adjacent pages go in one call (p.50-51, p.71-74, p.85, p.118: four calls). The model returns every principal risk on the pages with these fields: page, title, category, section, description, mitigation and confidence. Description and mitigation are copied verbatim if possible. However, this will not be possible if multiple principal risks exist within one paragraph or if one of the two fields is too long (above 15 sentences). The prompt includes my principal-risk definition, which separates risks *to* Vestas from Vestas' impacts *on* the world. Built: `notebooks/01_risk_extraction.ipynb` → `annual_report_reader/extraction/`.

**Step 2: merge, once per report.** The same risk appears on several pages, for example as a row in the p.71-74 table and as a full section on p.51. One call gets all the step-1 mentions and groups them. For each group it writes the final record: title, category, a 2-3 sentence description and the mitigation. This makes sure that the same risk is not mentioned as different risks over and over again which would create noise. The model only groups mentions and writes the text: the pages and sections of each record are copied in code from the mentions it chose, so they are always what step 1 extracted, and code checks that every mention is used exactly once (a mention the model leaves out becomes its own record, so nothing is lost). Built: `notebooks/02_risk_merge.ipynb` → `annual_report_reader/merge/`.

**Why two steps:**

- The two jobs are different. Step 1 finds and copies (if possible); step 2 de-duplicates and summarises.
- Each step can be evaluated on its own. If debugging is needed, this helps to locate the problem.

## Evaluation

**Golden set (hand-labelled).**

- Step 1: 14 risks across 8 pages. p.50 and p.72 are labelled as having *no* principal risks, to catch false positives.
- Step 2: 11 unique risks, each with every page it is mentioned on.

**Graders.** I used each grader for what it is good at (`notebooks/01b_extraction_eval.ipynb` and `02b_merge_eval.ipynb` → `annual_report_reader/evaluation/`). Both judges are Sol and grade `yes`/`no` only.

| What is graded | Step 1 | Step 2 |
| --- | --- | --- |
| Matching, title, description, mitigation | LLM judge (`extract-judge`), one call per page | LLM judge (`merge-judge`), one call per report |
| Section | LLM judge | |
| One-to-one pairs, valid ids | Code (invalid pairs are dropped with a reason) | Code |
| Category | Code | Code |
| Right mentions merged (pages), most informative first | | Code |
| Description ≤3 sentences, mitigation consistent with the mentions, title length | | Code checks |

**Consistency:** PLAN's second goal is that the same report gives the same risks every time, which is the "two analysts disagree" problem. The model isn't deterministic (the deployment ignores temperature), so I measure consistency directly: independent runs on the same pages, with nothing reused between runs. It's basic but it provides an idea of possible variance.

- In my tests the risks stay the same with minimal variance.
- The category is what moves. Given my definition of principal risk, it is expected.
- If we extend this to tougher material, I'm sure that could change.

**A realistic failure the eval catches.** I swapped the extraction model for a weaker, cheaper one (gpt-4.1-mini) and kept everything else the same (prompt, pages, judge):

The results fell quite sharply with GPT-4.1-mini. The evaluation indicates it clearly in my test both on recall and precision, depending on the run.

## Cost, latency and scaling

**Measured.**

| | Cost per report |
| --- | --- |
| Extraction, 8 pages (Luna, high effort) | ~$0.013 |
| Merge (Luna, medium effort) | ~$0.002 |
| Screening all 195 pages for risk pages (page locator, `notebooks/03`) | ~$0.09 |
| End to end | ~$0.10 |

That is ~$20 per quarter for 200 reports, against ~$1,333 of analyst time. Extraction and merge take about 25 seconds per report. Costs are token usage priced at $0.20 input / $1.20 output per 1M tokens (`price_in_per_m` / `price_out_per_m` in `ExtractionConfig` and `MergeConfig`); judge calls are not included.

**Scaling to many reports:**

- **Workers.** A queue feeds stateless workers, one report per job. The limit is the provider's rate limit, not compute.

- **Batch API.** If (my own) latency requirements can be relaxed the Batch API cuts cost by 50%.

- **Self-hosting** an open vision model on vLLM is only worth it if volume or data policy requires it. GPUs are not cheap, and the infra can take time to set up. Throughput scales with increased traffic.

- **Page screening.** Screening all pages costs about seven times the extraction itself. If we would like to extract more than just principal risks, embeddings would be interesting.

- **Consistency at scale.** Pin model versions. Keep prompts and golden sets versioned in git. Gate every prompt or model change on both accuracy and run-to-run agreement.
