"""Evaluation of step 1 (developed in notebooks/01b_extraction_eval.ipynb) and step 2 (notebooks/02b_merge_eval.ipynb).

Step 1, in data-flow order: `config` → `golden` → `coverage` → `judge` → `scores` → `runs`.
Step 2 reuses `config`, `golden`, `scores` and `runs`, with `merge_judge` and `merge_checks`.
"""
