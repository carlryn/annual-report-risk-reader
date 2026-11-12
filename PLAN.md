# PLAN

## Product context

Today, analysts read 200+ annual reports per quarter to track each company's principal risks. It is slow, and it is inconsistent: two analysts reading the same report often surface different risks.

Going through annual reports is a slow process. We can estimate the cost with a few assumptions.

About 20 minutes of analyst time per report at $20/hour. For 200 reports that is ~$1,333 per quarter, or ~$5,300 per year (assuming full automation). It does put a constraint on the system: it needs to be cheap to run to bring monetary value.

## Who is the user, and how do they consume the output?

The first user is the internal analyst. For each report, the pipeline proposes the principal risks, and the analyst reviews them in a UI on top of the risk database: accept, edit or reject. The UI needs to be built in a way that they can easily get an overview of the principal risks found in an annual report, view them in the PDF and decide.

The first slice doesn't replace the analyst; it speeds them up and collects review data. With that data, we can get a better idea on performance and build confidence. Then move to a fully automated solution.

## Product surface

A batch job, not an interactive tool. When a report is published (from a release calendar or an alert), its URL goes onto a queue. A worker(not human) downloads the PDF, extracts the principal risks and writes them to the database. I don't see latency mattering much here. This is not high frequency trading, and simply receiving the risks within a reasonable time frame should be enough. Let's say 30 minutes.

## Assumptions & in-scope

- I only process pages the case points to. I cover the two sections it asks to focus on: Risk management (p.50-51) and Material impacts, risks and opportunities (p.71-74). From the optional sections, I add Cyber security (p.118) and the first page of Climate change (p.85). Although I did run test with basic page classification in notebook 03.
- Reports vary from text-heavy to very graphical. The same risk can appear several times (CEO letter, risk section, appendix) and has to be merged into one. One paragraph can hold several risks, which then have to be split.
- I have created my own definition of what a "principal risk" is: [prompts/principal-risk-definition.md](prompts/principal-risk-definition.md).

## What to tackle later

- Finding the risk pages automatically in a ~190-page report. I also believe reading more context, and from other pages could be interesting. I did briefly start but didn't get very far. At least I could estimate the cost.
- It is important to understand that we are only looking at this report. I'm expecting this to be overfitted.
- I would potentially add a 3rd step. The step would aim at breaking up principal risks that could be multiple.
- Proper prompt management, experiment tracking.
- Better and more thorough testing.
- But in general, more and more reports.
