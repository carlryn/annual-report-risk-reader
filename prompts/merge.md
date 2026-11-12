You get the principal risks extracted page by page from one company annual report. Each item is a "mention" with an
id. The same risk is often mentioned in several places at different depth, for example a short row in a summary
table and a full section with mitigations elsewhere in the report.

Merge the mentions into a list of unique risks, and write the final record for each from all of its mentions.

When mentions are the same risk:

- Same risk means the same source of harm and the same effect on the company. Different wording, depth, section,
  page or category does not matter.
- Related but distinct risks stay separate, for example the same hazard for different affected groups, or two
  different causes listed under one heading.
- Every mention id must appear in exactly one unique risk. A mention with no duplicate is a unique risk on its own.
