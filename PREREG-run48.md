# run48: reverse dose 0.25 → j0+1 on one-step-early seeds. Pre-registered 2026-11-01 ~06:00, after run47, before training

**Why:**
- run47 B: 0.1 toward j0+1 flipped 0/3 early seeds.
- run44: 0.5 flipped 3/3.
- This run locates the reverse threshold on the same seeds, to test the asymmetry. Toward j0+2, 0.25 gave 3/3 (run46).

**Runs:** s47, s36, s44. BIASL1=0.25, OFF=1, steps < 2500, 8000 steps, RANDSTART=1 MLP=1 BIGRAM=0.5. Same code as before.

**Outcome:** same rules as PREREG-run47 (early / classic / mixed).

**Prediction:**
- ≥ 2/3 classic. P = 40%.
- If 0/3 or 1/3, the asymmetry stands at 0.25 too: the toward-early arm gave 3/3 at the same dose.
