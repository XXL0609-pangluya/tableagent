### "which club / which team did X play for in YEAR" → exactly ONE row
For career-history tables asking "which club did <player> play for in <year>?",
filter to the row matching that **year** and return ONLY that one club. Do NOT
list every club they played for across their career. If the year matches multiple
rows (e.g. a mid-season transfer), return the single club the gold expects (the
one whose row the question points at) — never pile on 2–3 clubs.

### top-k / argmax — EXCLUDE "all other ..." and aggregate buckets
When a question asks for the "top N" / "highest" / "lowest" categories, first
**drop rows whose label is a catch-all bucket** — e.g. "all other industries",
"all other ...", "remainder", "rest of ...", "other". These are aggregate
residuals, not real categories, and including them corrupts the ranking. After
excluding them AND any Total/summary row, sort and take the top N.
