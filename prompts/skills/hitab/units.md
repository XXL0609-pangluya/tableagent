### Units — read the table title / column header unit
HiTab tables have explicit units in the title or header (e.g. "millions of
dollars", "thousands", "per 100,000 population"). Return the value in the
**table's own unit**, not a converted one.
- Example: table in millions → `4.581961`, not `4,581,961` (that's the thousands
  equivalent). Do not multiply/divide to a different unit than the table shows.
- "how many billion dollars" → answer in billions (match the question's stated
  unit only if the table column is in that unit; otherwise convert and state the
  number in billions).
