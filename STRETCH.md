# STRETCH

What I'd build with one more week, in order.

1. **Find the risk pages automatically.** The page locator I started (`notebooks/03_page_locator.ipynb`) sends on only 17-25 of 195 pages as `likely`. That catches every golden page in some runs; in the latest run p.74 came back `possible` instead, so recall at the `likely` cut-off is not stable yet (`likely + possible` keeps every golden page, but sends on 79 pages). Next is to make it stable and plug it in front of the pipeline. I believe the text layer is enough to identify pages that could be interesting.
2. **Handle text-heavy reports.** My approach assumes graphical pages. For a page that is mostly running text (like a Word document), the text layer is reliable and much cheaper. A simple check on the text layer (how much text, how many columns) can route each page: text-heavy pages go through the text layer, graphical pages through images.
3. **Split mixed risks.** A third step that breaks up a risk which is really several risks in one paragraph. The merge has the opposite problem: it sometimes combines two risks the report lists side by side (the two p.118 cyber risks, in 3 of 3 runs when step 1 extracts both). A check in code that no record combines two mentions from the same page and section would catch it.
4. **Make the output answer the example queries.**
   - Tags instead of a single category. I can't even pick a category myself. Tags / multi-category makes much more sense.
   - A category taxonomy that works across companies.
   - A stable risk ID, so risks can be compared year over year and "newly elevated" risks found.
   - Confidence carried through to the final record, so analysts can filter on it. I wasn't able to get it working as I intended.
5. **Test more broadly.**
   - A golden set across several companies and years, to measure how overfitted the prompts are to Vestas.
   - A judge from another provider.
   - Test the merge on its own, fed the step-1 golden set, so a step-2 regression is not hidden behind step-1 misses.
   - Test the judges themselves: hand-written grader cases (a golden risk with one deliberate mistake, such as a description cut short, a wrong section or an opportunity passed off as a risk) and an identity check (each golden risk graded against an exact copy of itself), run against both judges.
   - How many adjacent pages can go in one call before quality drops. I sent p.71-74 together without a loss, but more pages could create issues.
