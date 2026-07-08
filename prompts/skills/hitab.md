# Skill: HiTab hierarchical-table QA

HiTab tables are statistical report tables with **hierarchical headers**. The
DataFrame you see is a *flattened* version of that hierarchy. Keep these in mind:

## THE ONE RULE: submit the VERBATIM output of the underlying computation
Every HiTab gold answer is the raw output of one spreadsheet formula over the
table cells. Match that output exactly — do NOT reformat it. Every convention
below is a corollary of this single rule:
- **A value you COMPUTE** (division, difference, sum, ratio, %-change): submit the
  raw float with **≥6 significant digits** (`0.186393`, `1.496599`) — never
  hand-rounded (`0.19`, `1.5`), never `%`-formatted.
- **A computed difference / decline / gap / "how many times"** is arranged to be
  **POSITIVE** — submit the positive magnitude (the direction is already in the
  question's words). Computed gold is essentially never negative.
- **A value you COPY from ONE cell**: submit it EXACTLY as shown — keep its own
  digits, its own scale, and its own sign (a cell showing `-2.27` or `52.1` stays
  `-2.27` / `52.1`).

## How the flattened table is laid out
- **Column names are hierarchical**, joined with ` | `. Example: a column shown
  as `custody | number` means the "number" sub-column under the "custody" group.
  Reference the FULL string exactly: `df['custody | number']`. Spaces around `|`
  matter.
- The **leftmost column** is the row header (e.g. `year`, `jurisdiction`,
  `province`). It is usually a KEY for filtering, rarely the answer itself.
- The **table title** in the schema is authoritative context (year, scope,
  unit). Use it: if the title says "2013/2014", the question's "in 2013/2014"
  is already the whole table — don't filter for it again.

## Answers are almost always NUMBERS — clean before arithmetic
- Cells carry commas (`116,442`), unicode minus (`−3`, `–5`), percent signs
  (`-12%`), and placeholders. Before any math, strip them:
  ```python
  s = df['col'].str.replace(',', '', regex=False
           ).str.replace('−', '-').str.replace('–', '-'
           ).str.replace('%', '', regex=False)
  num = pd.to_numeric(s, errors='coerce')
  ```
- Placeholders for **missing data** are `..`, `.`, `''`, `x`. Treat them as NaN,
  NOT as zero. `pd.to_numeric(..., errors='coerce')` handles this.

## "Total" rows — the #1 HiTab trap
- HiTab rows often include a **Total / Subtotal / "- total"** row that already
  aggregates the rows above it. If the question asks for a total, FIRST check
  whether a Total row already holds it — read that single cell instead of
  summing (summing would double-count).
- Conversely, when a question asks for a per-row value, **exclude** any Total
  row from your filter/sum.

## Aggregation types common in HiTab
- `sum` / "how many ... in total" over several rows → add the matching data
  cells (after stripping commas), BUT check for a Total row first.
- `div` / "what percent of X is Y" → `Y / X` (cast to float). Return the raw
  ratio the question asks for; do NOT multiply by 100 unless the question wants
  a percentage and the cell stores a decimal.
- `opposite` / "how many percent did X decrease" → the cell often stores the
  signed change (e.g. `-3`); the question asks for the magnitude `3`. Negate if
  needed: `answer = abs(num)` or `answer = -num`.
- percent change / rate columns: the value IS already a rate or percent change;
  return it directly, do not re-derive.

## Multi-answer questions
Some questions ask for several values in one sentence (e.g. "what are the
percent change from the previous year and from five years earlier, respectively?").
Return ALL of them as separate `items`, **in the order asked**. The gold is a
list matched as a set, but order them as the question lists them.

## HiTab answer conventions (measured on round 1 — these are NOT obvious; follow exactly)

### 0. NUMERIC PRECISION — never round a COMPUTED ratio/rate  [HIGHEST PRIORITY]
The scorer compares numbers with a VERY tight tolerance (~1e-4 relative). If you
COMPUTE a value (a division, ratio, "how many times", percentage, growth rate),
submit the **full-precision float — at least 6 significant digits** — do NOT round
to 1–2 decimals.
- `1.4966` is CORRECT; `1.5` is WRONG (0.2% off → fails the tolerance).
- `4.782609` is CORRECT; `4.78` is WRONG.
- `0.044185` is CORRECT; `0.0442` is WRONG.
- Practically: build the answer as `answer = [str(value)]` straight from the float,
  or `str(round(value, 6))` — never hand-round to 2 places. Do NOT format with `%`.
- This applies to ALL computed numbers (ratios, "times more likely", percentages,
  rates). It does NOT apply to a value you COPY verbatim from a cell — copy that
  exactly as shown (keep its own digits).

### 1. "what percent of X is Y" — DECIMAL RATIO vs PERCENTAGE (look at the source cells)  [HIGH PRIORITY]
Triggers: "how many percent of X is Y", "what percentage/percent of X is/was/are Y",
"what proportion of X", "what share of X", "X as a percentage of Y". The gold
depends on WHERE the numbers come from. **Decide by inspecting the source cells:**

**(a) If you are DIVIDING TWO RAW COUNTS** (e.g. 21704 people / 116442 people,
13097 officers / 68562 officers) → the gold is the **DECIMAL RATIO Y/X between 0
and 1**, NOT ×100.
- 21704 / 116442 → gold = `0.186393`, NOT `18.6`.
- Compute `answer = [str(Y / X)]`; do NOT multiply by 100; do NOT append "%".

**(b) If the table HAS a "percent" / "%" / "share" / "distribution" column whose
cells already hold PERCENTAGE values** (like 50, 25, 16, 6) → the gold READS or
SUMS those cells **as-is**. Do NOT divide by 100.
- "transition homes" cell = 50 → gold = `50`, NOT `0.5`.
- "emergency shelters" 25 + "women's emergency centres" 16 → gold = `41`, NOT `0.41`.

**How to tell**: look at the magnitudes. If the cells you're reading are raw counts
(tens/thousands/millions) and you're computing a ratio → return the decimal (a).
If the cells are already 0–100 percentages in a %-labeled column → return them
directly (b). When a table column header says "percent"/"%", prefer (b).

This rule does NOT apply to "how many PERCENTAGE POINTS did X change/decline" or
"how many percent did X decline/increase" — those are percentage-point magnitudes
and can be > 1 (e.g. `13`, `4`).

### 1b. growth rate / percentage change you COMPUTE → POSITIVE decimal
When you compute a percent change / growth rate as `(a - b) / c` → return the
**decimal between 0 and 1** (not ×100, not `%`), as a **positive** magnitude:
`|(308 - 373) / 373| = 0.174263`. Order the subtraction so the result is positive
— HiTab's *computed* changes are always positive. Only a value you READ verbatim
from a cell keeps a stored minus sign (see The One Rule).

### 2. "which group / which X" → return the table's SHORTEST canonical ROW LABEL
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

### 3. Multi-item answer formatting — split, don't duplicate, don't over/under-list
- **Split score/record pairs**: a "won-lost record" / score like `214-282` or
  `215-197` is TWO items `["214","282"]` / `["215","197"]`, not one hyphen string.
- **No duplicate items**: never return the same value twice. `["94.3","94.3"]` is
  WRONG — dedupe to `["94.3"]`. Before submitting, drop exact-duplicate items.
- **Match the asked count**: a question naming N categories wants exactly those N
  values (no extra, no missing). "top three" → exactly 3 items; a two-part
  "A and B respectively" → exactly 2 items in the asked order. Don't append an
  extra value and don't drop one of a hierarchical/multi-select set.

### 4. "A and B together / combined total" → sum into ONE item
If the question asks for a combined total of two named categories (e.g.
"emergency shelters AND women's emergency centres"), SUM their values into a
single item (`25 + 16 = 41` → `["41"]`), don't list them separately.

### 5. "how many TIMES higher/larger is X than Y" → X / Y (X is the bigger one)
`answer = [str(X / Y)]` where X is the larger value. Mind numerator/denominator:
"how many times higher is A than B" = A / B.

### 6. SIGN — computed ⇒ POSITIVE magnitude; copied cell ⇒ keep the cell's sign  [HIGH PRIORITY]
The sign is decided by HOW you got the number, not by re-reading the question:
- If you **computed** the change / decline / difference / gap (a subtraction,
  division, or negating a stored value) → submit the **POSITIVE magnitude**.
  HiTab's computed gold is never negative — the direction already lives in the
  question's words ("decline", "decrease", "gap"). Apply `abs(...)` before
  submitting.
  - "what was the decline in cases ..." (cell `-24`) → `24`, never `-24`.
  - "percentage-point loss ..." → `2.1`, never `-2.1`; "the gap was 8.1" → `8.1`.
  - computed growth/percent change → positive decimal (see §1b).
- If you **read ONE cell verbatim** and that cell itself already shows a minus (a
  stored percent-change / difference like `-2.27`, `-23.4`) → keep it EXACTLY,
  including the sign.

When in doubt for anything you COMPUTED, use the positive magnitude.

### 7. Units — read the table title / column header unit
HiTab tables have explicit units in the title or header (e.g. "millions of
dollars", "thousands", "per 100,000 population"). Return the value in the
**table's own unit**, not a converted one.
- Example: table in millions → `4.581961`, not `4,581,961` (that's the thousands
  equivalent). Do not multiply/divide to a different unit than the table shows.
- "how many billion dollars" → answer in billions (match the question's stated
  unit only if the table column is in that unit; otherwise convert and state the
  number in billions).

### 8. "which club / which team did X play for in YEAR" → exactly ONE row
For career-history tables asking "which club did <player> play for in <year>?",
filter to the row matching that **year** and return ONLY that one club. Do NOT
list every club they played for across their career. If the year matches multiple
rows (e.g. a mid-season transfer), return the single club the gold expects (the
one whose row the question points at) — never pile on 2–3 clubs.

### 9. top-k / argmax — EXCLUDE "all other ..." and aggregate buckets
When a question asks for the "top N" / "highest" / "lowest" categories, first
**drop rows whose label is a catch-all bucket** — e.g. "all other industries",
"all other ...", "remainder", "rest of ...", "other". These are aggregate
residuals, not real categories, and including them corrupts the ranking. After
excluding them AND any Total/summary row, sort and take the top N.

## Recommended sequence
1. `inspect_table` — read the title, the hierarchical column names, and sample
   values. Notice any Total row.
2. `search_columns` / `search_cells` — ground the question's words to the real
   (hierarchical) column names and row-label values.
3. `run_python` — cast to numeric, filter the right rows (watch Total rows),
   compute, and set `answer` (a list). Set `evidence` with the row labels /
   column used.
4. `submit_answer` immediately once `answer` is set.
