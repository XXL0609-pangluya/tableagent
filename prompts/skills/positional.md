# Skill: positional (next / previous / before / after / first / last / middle)

Applies when the question is about ORDER or position in the table:
"next", "after", "previous", "before", "preceding", "following", "above",
"below", "first", "last", "in the middle of A and B", "consecutive".

## FIRST — pick the right ordering axis (DO NOT default to row +/- 1)
"before / after / previous / next" is almost NEVER physical row +/- 1. It means the
neighbour along the axis the question orders by. Decide which axis FIRST:
- **Value/time order (the common case)**: if there is a Year / Date / Edition / Rank /
  Season column, "before/after/previous/next" is defined by THAT column's VALUES.
  "won after Ballymore Eustace" = the team whose Year is the next one later;
  "signed before Troy Nolan" = the latest Date that is earlier than Troy Nolan's.
- **Listed order**: ONLY when the question literally says "listed", "in the table",
  "next row", "as shown".
- **Ranked order**: "next highest scorer after X" = sort by points, take the neighbour.

## CRITICAL: the table may be sorted DESCENDING — row order != time order
Many WTQ tables list newest-first (Year 2011, 2010, 2009 ... going DOWN). So the row
BELOW an anchor is EARLIER in time, not later. NEVER assume `pos + 1` = "after / later".
Always derive the neighbour from the SORTED VALUES and PRINT them to confirm:

```python
import re
df = df.reset_index(drop=True)
df['_year'] = df['Years won'].astype(str).str.extract(r'(\d{4})').astype(float)
anchor = df.loc[df['Team'].str.contains('Ballymore Eustace', case=False), '_year'].iloc[0]
# "after" = next LARGER value; "before" = next SMALLER value
later  = df[df['_year'] > anchor].sort_values('_year').head(1)   # after
earlier= df[df['_year'] < anchor].sort_values('_year').tail(1)   # before
print('anchor=', anchor)
print('after  ->', later[['Team','_year']].to_dict('records'))
print('before ->', earlier[['Team','_year']].to_dict('records'))
answer = later['Team'].tolist()        # pick after/before per the question
```

### "signed / born / released / joined ... before/after X" → SORT BY THE DATE
When the question links order to an EVENT ("signed before", "released after", "joined
prior to"), the order is the DATE column — even if the rows look ordered by something
else (position, name) and even if dates have NO year (parse month names). Do NOT use
row position here.
```python
import pandas as pd
d = pd.to_datetime(df['Date signed'], errors='coerce', format='mixed')  # "March 29" etc.
df = df.assign(_d=d)
anchor = df.loc[df['Player'].str.contains('Troy Nolan', case=False), '_d'].iloc[0]
before = df[df['_d'] < anchor].sort_values('_d').tail(1)   # latest date BEFORE anchor
print(before[['Player','Date signed']].to_dict('records'))
answer = before['Player'].tolist()
```

If the axis truly is listed order, anchor by position instead:
```python
pos = df.index[df['Player'].astype(str).str.contains('Troy Nolan', case=False)][0]
answer = df.loc[pos - 1, 'Player']     # previous-listed (pos + 1 = next-listed)
```

- "in the middle of A and B": rows between the positions/values of A and B.
- "first/last item": first/last DATA row, but FIRST drop any `Total`/summary row.
- Return the value from the COLUMN THE QUESTION ASKS ABOUT. "what ROUND came next"
  -> read the `Round` column of the neighbour row, not an adjacent column.

## "name another X (other than Y)" → return exactly ONE
"name another region other than Greece", "a different team than X" want a SINGLE
other item (usually the first one that isn't the excluded Y), not the whole list.
Return one item; do not dump every remaining row.

## Mistakes to avoid
- Defaulting to `pos +/- 1` when a Year/Date/Rank column defines the real order.
- Assuming row order = chronological order. Tables are often DESCENDING — print the
  ordering column to see the direction before you trust "next row = later".
- Reading the wrong column off the neighbour row (return the asked-for column).
- Returning the `Total`/summary row as the "last" entry (or as first/most/least).
  A summary label like `Total`/`Average` is NEVER a valid entity answer — drop those
  rows first, then take the position.
- Off-by-one: confirm which direction "before/after" means for this table.
- "name another X" → returning several items instead of one.
