You compare the unique risks produced for one annual report (PREDICTED) with a human-labelled reference list for
the same report (GOLDEN). Each risk has an id, a title, a description, a mitigation and the pages it is mentioned on.
The predicted descriptions and mitigations are summaries written from several places in the report, so their
wording will differ from the golden ones.

Step 1: match. Pair each golden risk with the predicted risk that describes the same underlying risk.
Use the title, description and pages to decide; match on meaning, not on words.
- Each golden risk and each predicted risk can be in at most one pair.
- If a predicted risk combines two golden risks, pair it with the closest one only and leave the other golden
  risk unpaired.
- If two predicted risks describe the same golden risk, pair the closer one only and leave the other unpaired.
- Leave a risk unpaired when nothing is a reasonable match. Do not force pairs.
- A pair is about being the same risk, not about how well it was written: a vague description or a missing
  mitigation is still the same risk, and is graded in step 2.

Step 2: grade each pair, yes or no.
- title: yes = a reader would recognise it as the same risk; own wording is fine. no = clearly broader or
  narrower, or names another risk.
- description: yes = the main claim of the golden description is there (what the risk is and how it could affect
  the company), and nothing contradicts the golden. Extra detail is fine, also detail that comes from another
  golden risk the prediction combined. no = the main claim is missing, too vague to recognise, or contradicted.
- mitigation: yes = the main mitigation actions of the golden are there, or both state none ("None"). Fewer words
  are fine. no = a main action is missing or different, or only one of them states a mitigation.
Give a one-sentence reason per pair.
