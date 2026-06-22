# Skill: positional (next / previous / before / after / first / last / middle)

Applies when the question is about ORDER or position in the table:
"next", "after", "previous", "before", "preceding", "following", "above",
"below", "first", "last", "in the middle of A and B", "consecutive".

## FIRST — pick the right ordering axis
- **Listed order** (table row index): "next to X in the table", "last listed".
- **Ranked order** (sort by a value column): "next highest scorer after X",
  "top scorer next to Merritt" (= second-highest points, not adjacent row).
- **Time/date order** (sort by date column): "signed before X", "previous player
  signed" — use `Date signed`, NOT row position.

When unsure, inspect both axes before submitting.

## Use ROW POSITION relative to a matched row — do not eyeball
Keep the table's listed order (do NOT sort) unless the question clearly asks for a
ranking ("the next heaviest" = sort by weight; "the next opponent listed" = listed
order). Anchor on the row that matches the named entity, then move by position.

```python
df = df.reset_index(drop=True)               # 0..n-1 positional index
pos = df.index[df['Player'].astype(str).str.contains('Troy Nolan', case=False)][0]
answer = df.loc[pos - 1, 'Player']           # previous / before  (pos + 1 for next / after)
```

- "next/after X (as listed)": row at `pos + 1`.
- "previous/before X (as listed)": row at `pos - 1`.
- "in the middle of A and B": the row(s) between the positions of A and B,
  e.g. `mid = (posA + posB) // 2; answer = df.loc[mid, 'Col']`.
- "first/last listed item": first/last DATA row — but first EXCLUDE any
  `Total`/summary row (see the aggregation skill) so "last" isn't the totals row.
- "the next heaviest/fastest/... after X": sort by that numeric column, find X's
  position in the sorted order, take the neighbour.

## Mistakes to avoid
- Returning the `Total`/summary row as the "last" entry.
- Sorting when the question means listed order (or vice versa).
- Off-by-one: confirm which direction "before/after" means for this table.
