# Skill: HiTab hierarchical-table QA

HiTab tables are statistical report tables with hierarchical headers, shown to you
as a *flattened* DataFrame.

## Table layout
- Column names are hierarchical, joined with ` | ` (e.g. `custody | number` = the
  "number" sub-column under "custody"). Reference the FULL string; spaces matter.
- The leftmost column is the row header (year, jurisdiction, ...): a KEY for
  filtering, rarely the answer itself.
- The table title is authoritative context (year, scope, unit). If the title
  already says "2013/2014", don't filter the question's "in 2013/2014" again.

## Before arithmetic
- Cells carry commas (`116,442`), unicode minus (`−`, `–`), `%`. Strip them, then
  `num = pd.to_numeric(s, errors='coerce')`.
- Missing-data placeholders (`..`, `.`, `''`, `x`) → NaN, NOT zero.
- Total / Subtotal / "- total" rows already aggregate the rows above. For a total,
  READ that row (re-summing double-counts); for per-row values, EXCLUDE it.

## Answer conventions (HiTab-specific — follow exactly)

**0. PRECISION [highest].** A COMPUTED number (ratio / percent / "times" / rate)
must be the full float, **≥6 significant digits** — `1.4966` not `1.5`, `0.044185`
not `0.0442`. Use `str(value)` or `str(round(value, 6))`, never hand-round, no `%`.
A value COPIED verbatim from a cell keeps that cell's exact digits.

**1. "percent / proportion / share of X" — decimal vs percentage by SOURCE.**
- (a) dividing two RAW COUNTS → DECIMAL ratio in 0–1 (`21704/116442 → 0.186393`);
  no ×100, no `%`.
- (b) reading/summing cells already in a `percent`/`%`/`share` column → keep them
  as-is (`50`, `41`); do NOT ÷100.
- Not for "percentage points … change/decline" (magnitudes, can be >1).
- growth rate `(new-old)/old` → signed decimal (`-0.174263`), no ×100.

**2. "which group / region / sector"** → the table's SHORTEST canonical row label,
not the question's wording (`transgender` not "transgender canadians"; `aboriginal`
not "aboriginal women"). Read the label column of the matched row; strip qualifiers
the label itself doesn't contain.

**3. Multi-item formatting.** Split score/record pairs (`215-197` → `["215","197"]`);
NO duplicate items (dedupe exact repeats, `["94.3","94.3"]` → `["94.3"]`); return
exactly the N values asked — no extra, none missing — in the order asked.

**4. "A and B combined / together / total"** → SUM into ONE item (`25+16 → ["41"]`).

**5. "how many TIMES higher/larger is A than B"** → `A / B` (mind numerator order).

**6. SIGN — judge from meaning, not a formula.** State the number as the answer
sentence would read. change / difference / gap / "by how much did X decrease" is
usually the POSITIVE magnitude (`12`, not `-12`). Keep a minus sign only for a
neutral signed *net* value (net change, growth rate). When unsure, prefer the
positive magnitude.

**7. Units = the table's own unit** (from title/header). A millions table →
`4.581961`, not its thousands form. Don't convert to a unit the table doesn't use.

**8. "which club/team did X play for in YEAR"** → filter to that year, return
exactly ONE club, not the whole career.

**9. top-k / argmax** → first DROP catch-all buckets ("all other …", "rest of …",
"other") and Total rows, then rank and take the top N.

## Sequence
`inspect_table` (title, hierarchical columns, any Total row) → `search_columns` /
`search_cells` (ground question words to real names) → `run_python` (clean, filter,
compute, set `answer` list + `evidence`) → `submit_answer`.
