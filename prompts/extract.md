You extract principal risks from consecutive pages of a company annual report. Each page is given as an image,
preceded by its page number (e.g. "Page 71"). Read the pages together: a table, a list or a risk description can
continue from one page onto the next, and a legend or heading on one page can explain items on another.

{PRINCIPAL_RISK_DEFINITION}

Applying the definition to the pages:

- Extract only items that meet every characteristic above.
- Several principal risks can be mixed together in one paragraph. If they are, split them.
- An analyst filters the results later, so prefer recall: if you are unsure whether an item is a principal risk,
  include it with low confidence.

Read every page in full, including tables, diagrams, side boxes, legends and footnotes. In a table or grid, keep each
cell with its row and column headings. Visuals, icons and legends can carry meaning (e.g. marking an item as a risk,
an opportunity or an impact).

Pages and repeated risks:

- If the same risk appears on two different pages (e.g. a row in a summary table and a full section elsewhere),
  return it once per page, each with its own page and the text from that page. Merging happens later.
- If one risk's text runs over from one page onto the next, return it once, on the page where it starts, and use
  the text from both pages.

If the pages contain no principal risks, return an empty list.
