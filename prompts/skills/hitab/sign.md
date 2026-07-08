### SIGN of the answer — decline/loss/less ⇒ POSITIVE magnitude by default  [HIGH PRIORITY]
Decide the sign by what the question MEANS — but there is a strong default you must
follow unless the wording clearly overrides it.

**DEFAULT (the common case): report the POSITIVE magnitude — strip any minus sign.**
If the question's own words already carry the direction —
"**decline / declined / decrease / drop / fell / loss / lost / less / lower /
reduction / gap / difference / by how much did X change**" — then the answer is the
size of that change as a **positive** number. The direction is in the sentence, not
in the number.
- "what was the percent **declines** in cases ..." cell `-24` → answer `24`, NOT `-24`.
- "percentage-**point loss** ..." → `2.1`, NOT `-2.1`.
- "cases **decreased** 12%" → `12`. "the **gap** was 8.1 points" → `8.1`.
- So after you compute the change, apply `abs(...)` for these decline/loss/gap/less
  phrasings before submitting.

**EXCEPTION — keep the real sign only for a NEUTRAL signed/net quantity** whose
direction is genuinely unknown from the wording and could go either way:
- "what is the **net change** / **growth rate** / **percent change**" with NO
  decline/loss word → a real decrease stays NEGATIVE (e.g. `(308-373)/373 = -0.174263`).
- A raw signed difference explicitly framed as "how many **more/fewer**" may keep its
  sign if the question wants the signed gap; but "how many did X **decrease by**" →
  positive magnitude.

When in doubt for a decline/loss/gap/difference question, choose the **positive
magnitude** — that is what HiTab's gold uses far more often.
