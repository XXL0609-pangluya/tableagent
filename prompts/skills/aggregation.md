# Skill: aggregation (count / sum / difference / extremes)

Applies when the question counts, sums, averages, takes a difference, or asks for
a most/least/highest/lowest value.

## Before aggregating — confirm what "total" means
- **Row count vs unique names**: "how many sheriffs in total" usually counts every
  tenure/appearance (76 entries), NOT unique people (66).
- **Team Totals row vs sum of players**: "total assists and turnovers combined"
  may need the official **Team Totals** row (includes team-only stats), not a
  manual sum of player rows.
- Still exclude junk summary labels when summing *player* rows, but check whether
  the question wants the precomputed totals row instead.

## CRITICAL: exclude summary / total rows first (when summing player rows)
Many WTQ tables end with a summary row such as `Total`, `Totals`, `TOTAL`, `Sum`,
`All`, `Overall`, `Average`, or a row whose label cell is blank/`-`. Including it
double-counts and is the #1 cause of wrong sums/counts/extremes.

Before aggregating, drop those rows:
```python
label = df.columns[0]                     # the row-label column (usually the first)
junk = {'total', 'totals', 'sum', 'all', 'overall', 'average', 'averages', '', '-', '—'}
data = df[~df[label].astype(str).str.strip().str.lower().isin(junk)]
```
Sanity check: if a single value (e.g. a "Total" cell) equals the sum of the
others, you almost certainly included a summary row — recompute on `data`.

## Patterns (operate on `data`, not `df`)
- count rows matching a condition:
  `answer = int((data['Col'] == value).sum())`
- "how many X" by a category value: filter then count rows.
- sum / combined total of one or more numeric columns:
  `s = data['Col'].str.replace(',', '', regex=False).astype(float); answer = int(s.sum())`
- difference between the highest and lowest:
  `s = data['Pts'].str.replace(',', '', regex=False).astype(float); answer = int(s.max() - s.min())`
- most / least of a numeric column:
  `s = data['Pts'].str.replace(',', '', regex=False).astype(float); answer = data.loc[s.idxmax(), 'Name']`

## Mistakes to avoid
- Counting/summing the Total row (recompute on `data`).
- Forgetting to strip ',' / units / '%' before casting to number.
- Counting blank rows or sub-header rows.
