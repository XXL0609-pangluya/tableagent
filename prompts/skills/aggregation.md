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

## CRITICAL: "which/who had the most/least X" → return the ENTITY NAME
After `idxmax`/`idxmin` you have the right ROW. Now return the column the question
asks about — the descriptive **name** (Artist, Team, Player, Owner...), NOT a numeric
id (Draw, Rank, No., #) and NOT the measure value (the points/score itself).
```python
i = s.idxmin()
answer = data.loc[i, 'Artist']   # the name the question wants
# NOT data.loc[i, 'Draw']  (that's just the row id)  — NOT s.min()  (that's the value)
```
Only return a number here if the question explicitly asks for one ("which YEAR/NUMBER
had the most ...").

## Counting YEARS / time spans — subtract real values, never row positions
For "how many years before/since/between ...", compute from the actual Year/Date
**values**, not the row index. The row index is a position, not a count of years
(rows can skip years, and 0-based indices are off by one).
```python
start = int(df['Year'].iloc[0])
first_pf = int(df.loc[df['Status'] == 'Partly Free', 'Year'].iloc[0])
answer = first_pf - start          # years before the first Partly Free status
# NOT df[df['Status']=='Partly Free'].index[0]  (that's a row position)
```

## Clean dirty values BEFORE comparing or summing (very common trap)
Numbers are often wrapped in junk: units, parentheses, percent signs, a unicode
minus `−` (U+2212, NOT the ascii `-`), thousands separators. Comparing these as
strings gives wrong results (e.g. counting "scores less than -14" on raw text).
Extract and normalize first:
```python
s = (df['Winning score']
     .str.extract(r'([-−–]?\s*[\d.,]+)')[0]   # leading number, allow unicode minus
     .str.replace('−', '-', regex=False)       # normalize minus
     .str.replace(',', '', regex=False)
     .astype(float))
answer = int((s < -14).sum())
```

## Duplicate / split columns (side-by-side sub-tables)
If inspect_table warns of columns like `Position` AND `Position.1`, the table is two
sub-tables placed side by side. To count/aggregate the LOGICAL column, combine them:
```python
combined = pd.concat([df['Position'], df['Position.1']], ignore_index=True)
answer = int((combined.str.strip().str.lower() == 'philanthropist').sum())
```

## Verify a count by printing the matched rows
For counts, print what you matched and eyeball it before trusting `len()`:
```python
hits = data[data['Col'].str.contains('philanthropist', case=False, na=False)]
print(hits[['Col']]); answer = int(len(hits))
```

## Mistakes to avoid
- Counting/summing the Total row (recompute on `data`).
- Forgetting to strip ',' / units / '%' / unicode minus before casting to number.
- Counting blank rows or sub-header rows.
- Counting only one of two duplicate/split columns (combine them first).
- For "which had the most/least X", returning the numeric id (Draw/Rank/No.) or the
  measure value instead of the entity NAME the question asks for.
- Using a row index/position as a number of years or items (subtract real values).
