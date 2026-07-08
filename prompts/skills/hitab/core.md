# Skill: HiTab hierarchical-table QA

HiTab tables are statistical report tables with **hierarchical headers**. The
DataFrame you see is a *flattened* version of that hierarchy. Keep these in mind:

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

## NUMERIC PRECISION — never round a COMPUTED ratio/rate  [HIGHEST PRIORITY]
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

## Recommended sequence
1. `inspect_table` — read the title, the hierarchical column names, and sample
   values. Notice any Total row.
2. `search_columns` / `search_cells` — ground the question's words to the real
   (hierarchical) column names and row-label values.
3. `run_python` — cast to numeric, filter the right rows (watch Total rows),
   compute, and set `answer` (a list). Set `evidence` with the row labels /
   column used.
4. `submit_answer` immediately once `answer` is set.
