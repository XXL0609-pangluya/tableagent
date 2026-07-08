### Multi-item answer formatting — split, don't duplicate, don't over/under-list
- **Split score/record pairs**: a "won-lost record" / score like `214-282` or
  `215-197` is TWO items `["214","282"]` / `["215","197"]`, not one hyphen string.
- **No duplicate items**: never return the same value twice. `["94.3","94.3"]` is
  WRONG — dedupe to `["94.3"]`. Before submitting, drop exact-duplicate items.
- **Match the asked count**: a question naming N categories wants exactly those N
  values (no extra, no missing). "top three" → exactly 3 items; a two-part
  "A and B respectively" → exactly 2 items in the asked order. Don't append an
  extra value and don't drop one of a hierarchical/multi-select set.

### "A and B together / combined total" → sum into ONE item
If the question asks for a combined total of two named categories (e.g.
"emergency shelters AND women's emergency centres"), SUM their values into a
single item (`25 + 16 = 41` → `["41"]`), don't list them separately.
