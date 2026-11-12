# Prompts

One file per step; the file name is what the configs name (`prompt` in `ExtractionConfig` and `MergeConfig`,
`judge_prompt` in `JudgeConfig`, `PROMPT` in notebook 03). Edit a prompt in place: git keeps its history, and every
saved run stores the exact system prompt it sent, so a result can always be traced to its prompt.

The output schema is part of the instructions: with structured output the model sees each field's description.
Rules about one field live there (`extraction/schema.py`, `merge/schema.py`); the prompt holds the task and the rules that span
fields. Extraction and merge runs save the schema next to the system prompt (`output_schema`).

| File | Used by | Step |
| --- | --- | --- |
| `extract.md` | `01_risk_extraction.ipynb`, `extraction/` | principal risks from page images (several adjacent pages per call) |
| `extract-judge.md` | `01b_extraction_eval.ipynb`, `evaluation/judge.py` | step-1 judge: match, then title, section, description, mitigation |
| `merge.md` | `02_risk_merge.ipynb`, `merge/` | merge duplicate mentions and write the final records |
| `merge-judge.md` | `02b_merge_eval.ipynb`, `evaluation/merge_judge.py` | step-2 judge: match, title, description, mitigation |
| `page-locator.md` | `03_page_locator.ipynb` | screen windows of pages: which pages could contain principal risks |

`{PRINCIPAL_RISK_DEFINITION}` (extraction and page-locator prompts) is replaced with
`principal-risk-definition.md` in this folder, so a change to the definition and a change to the prompt stay separate.
