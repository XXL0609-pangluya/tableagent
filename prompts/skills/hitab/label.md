### "which group / which X" → return the table's SHORTEST canonical ROW LABEL
For "which group of people ... ?", "which region ... ?", "which sector ... ?"
questions, the gold is the **canonical row label as it appears in the table**,
which is often SHORTER or different from the phrase used in the question.
- Example: question says "transgender canadians or cisgender canadians?" → gold
  is `transgender` (the short table label), NOT "transgender canadians".
- Example: question says "sexual minority canadians with a disability" → gold is
  `person with disability` (the table's canonical axis label).
- **Strip qualifiers the label doesn't include.** If the matching axis label is
  just `aboriginal`, return `aboriginal` — NOT `aboriginal women` / `aboriginal
  men` / `aboriginal females` (do not append the sex/age qualifier from the
  question or a neighbouring header). If the label is `fair or poor`, return
  `fair or poor`, NOT "those in fair or poor health".
- Rule: after finding the matching row, return the value of the row-header /
  label column for that row (the SHORTEST canonical form), not the question's
  own wording. If unsure which label, dump the candidate row and read its label
  column exactly.
