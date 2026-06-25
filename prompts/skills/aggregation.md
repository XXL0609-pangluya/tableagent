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

Prefer the human-readable TITLE/NAME column, not a code/reference/abbreviation. E.g.
"which card was issued most" → the card's name (`Royal Wedding (The Princess Anne)`),
NOT its catalogue ref (`PHQ 4`). If several columns could be "the name", pick the
descriptive one, not the short id/code.

## Comparative / superlative words: decide what they mean IN THIS TABLE first
"higher / better / top / most / longer / leading" all mean "a BETTER result" — but what
counts as "better" depends on the column you measure and how THIS table is built. Do not
assume a fixed direction. Before computing, ask:
- Which column does the word refer to? ("higher rank" → the Rank/Position; "most points"
  → the points column; "longest term" → a duration you may have to compute from dates.)
- For that column, does a BIGGER number mean better, or SMALLER? A Rank/Position of 1 is
  the TOP, so "higher/better rank" = the SMALLEST rank number. Points/score: bigger = better.
- Is the table already SORTED by result, with the leading rows' rank cell left BLANK? Then
  a blank rank is the BEST, not the worst — row position is the real ranking.
Read a few rows and state the mapping ("here higher rank = smaller number = earlier row")
before you pick `idxmax`/`idxmin`/position. Then verify by printing the candidate rows.

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

## Verify a count by printing the matched rows — then SCAN the list like a human
For counts, ALWAYS print the matched VALUES (not just the number) and read the list
before trusting `len()`. A quick scan catches the two most common counting traps:
```python
hits = data[data['Col'].str.contains('philanthropist', case=False, na=False)]
print(hits['Col'].tolist()); answer = int(len(hits))
```

### Trap 1 — the same ENTITY appears on several rows (count DISTINCT, not rows)
When you count real-world ENTITIES that can recur — people, countries, teams, clubs —
the same one often spans several rows (multiple terms, seasons, years). Scan the printed
names: if any repeat, you almost certainly want DISTINCT, not row count.
```python
names = df[df['Name'].str.startswith('John')]['Name'].tolist()
print(names)                          # e.g. ['John T. Jordan','John T. Jordan','John Collins',...]
answer = df[df['Name'].str.startswith('John')]['Name'].nunique()   # distinct people
```
(But when you count EVENTS/occurrences — games, appearances, matches, wins — repeats are
real and row count is correct. Decide which one the question asks for.)

### Trap 2 — the same thing is written several ways (don't miss VARIANTS)
Exact `==` misses spelling variants: accents ("Salome" vs "Salomé"), suffixes
("Salome" vs "Salome, Op. 55"), trailing qualifiers. So BEFORE counting a text label,
FIRST print the column's UNIQUE values to see every way the target is written — the
missing variants will NEVER show up if you only look at the rows you already matched:
```python
print(df['Title'].unique())     # reveals 'Salomé', 'Salome, Op. 55', 'Salome, Op. 19' ...
```
Then match by normalized contains, not `==`:
```python
import unicodedata
def norm(s): return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c)).lower()
titles = df['Title'][df['Title'].apply(lambda t: norm(t).startswith('salome'))]
print(titles.tolist()); answer = int(len(titles))
```

## Mistakes to avoid
- Counting/summing the Total row (recompute on `data`).
- Forgetting to strip ',' / units / '%' / unicode minus before casting to number.
- Counting blank rows or sub-header rows.
- Counting only one of two duplicate/split columns (combine them first).
- For "which had the most/least X", returning the numeric id (Draw/Rank/No.), a
  code/reference (PHQ 4), or the measure value instead of the entity NAME asked for.
- Treating "higher/better rank" as a larger number — rank 1 is the TOP.
- Using a row index/position as a number of years or items (subtract real values).
