### "what percent of X is Y" — DECIMAL RATIO vs PERCENTAGE (look at the source cells)  [HIGH PRIORITY]
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

### growth rate / percentage change computed as (new - old) / old → SIGNED DECIMAL
"what is the growth rate ... from A to B" / "how many percentage points did X
decline down from FY" where you compute `(new - old) / old` → return the **signed
decimal**, e.g. `(308 - 373) / 373 = -0.174263`. Do NOT multiply by 100 (NOT
`-17.4`), and keep the sign.
