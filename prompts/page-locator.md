You screen consecutive pages of a company annual report to find the pages that could contain the company's
principal risks. A later step reads every page you flag in detail and extracts the risks, so a missed page is worse
than a page flagged too many. Flag a page only for what is on it, though: the neighbouring pages are context, not
evidence.

Each page is preceded by its page number (e.g. "Page 51"). Use those numbers, not page numbers printed on the page.

{PRINCIPAL_RISK_DEFINITION}

For every page you are given, return exactly one entry with:

- page: the page number given before the page.
- verdict:
  - confident: the page itself presents one or more specific risks that meet the definition. Examples: a list or grid
    of main, key or principal risks; a table of material risks (including tables that mark items as risks, impacts
    or opportunities, if at least one item is a risk to the company); a section dedicated to one specific risk
    (e.g. cyber security or climate-related transition risk) with its description or mitigation.
  - possible: the page names specific risks to the company, but it is unclear whether they meet the definition.
    Examples: risks the company may present as minor or immaterial; risks named in an outlook or strategy section;
    financial risk notes (currency, interest rate, credit, liquidity); risk text that continues from the previous
    page.
  - unlikely: the page names no specific risk to the company. Examples: table of contents; descriptions of how risk
    is managed that do not name the risks; impacts the company has on people or the environment only;
    opportunities only; passing mentions of the word "risk"; financial statements without risk discussion.
- reason: one sentence on what on the page supports the verdict (name the heading or table if there is one).
- candidate_risks: short names of the risks on the page that could be principal risks. Empty for unlikely pages.
- section: the section the page belongs to, as printed: the heading at the top of the page, or the running section
  name in the page header if the page has no heading of its own. Prefer the most specific heading that covers the
  whole page (e.g. "Material impacts, risks and opportunities", not "Sustainability statement").
- continues_previous: true if the page continues the same section, table or list as the page given just before it
  (the same heading carries on, or a table or list runs over the page break). False if a new section starts at the
  top of the page. For the first page you are given, you cannot see the page before it: return false.

Read every page in full, including tables, diagrams, side boxes and legends. Give every page a section and a continues_previous
value, whatever its verdict. Return the entries in page order.
