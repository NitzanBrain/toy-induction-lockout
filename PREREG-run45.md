# run45: control for run44 (pre-registered 2026-10-31 ~01:55, before training)
**Question:** does any +0.5 push in layer 1 make classic seeds go one step early, or is it specific to key j0+2?
**Control:** BIASL1=0.5, BIASL1_OFF=3 (key j0+3, two steps early, an irrelevant target), steps < 2500, 8000 steps, on the same classic seeds s39, s45, s40. Otherwise identical to run44.
**Outcome (frozen):** run43 rule (some L1 head with j0+2 ≥ 0.50 and j0+1 ≤ 0.10 → one step early). I also report the max L1 mass on j0+3 (computed with the same code path as analyze_qkv; new script analyze_off3.py, written before seeing results).
**Prediction:** ≤ 1/3 become one step early (P = 70%). If ≥ 2/3 do, run44's specificity claim is weakened: the push would then be generic disruption, not the j0+2 preference.
