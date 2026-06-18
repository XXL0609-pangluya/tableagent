# Skill: general table QA

Applies to any question when no more specific skill is selected.

## Recommended tool sequence
1. `inspect_table` — see columns, types, sample values.
2. If the question mentions an entity/value/column that may not match exactly,
   use `search_columns` and/or `search_cells` to ground it.
3. `run_python` — write pandas that sets `answer`. Patterns:
   - lookup:      `answer = df.loc[df['Col'] == value, 'Target'].tolist()`
   - count:       `answer = int((df['Col'] == value).sum())`
   - superlative: cast then idxmax/idxmin, e.g.
                  `s = df['Depth'].str.replace(',', '').astype(float); answer = df.loc[s.idxmax(), 'Name']`
   - difference:  cast two numbers, subtract, `answer = abs(a - b)`
   - previous/next: sort or use position relative to the matched row.
4. Check the result is non-empty and sensible; if not, adjust and retry.
5. `submit_answer` with the final items.

## Common mistakes to avoid
- Forgetting to strip thousands separators / units before casting to number.
- Returning the whole row instead of the asked-for column.
- Returning multiple items when one is expected (or vice versa).
- Answering from the schema/sample without checking the actual rows.
