# run50: data order on BORDERLINE inits (pre-registered 2026-11-02 ~06:45, after run49, before any run50 training)

**Why.** run49 kept 11/12 routes, but used the clearest inits. The question is whether batch order can tip a close call.

**Inits (run43 route, binary rule early vs not-early):** s46 (early, mid marker rank), s35 (early, mid rank), s32 (not-early, "mixed": one L1 head on j0+2 at 0.68), s33 (not-early, mixed, 0.62).

**Runs.** Same as run49 (run50-dataorder.sh is run49's script with these seeds): DATA_SEED 1001–1003, HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5, 8000 steps, no bias; analyze_qkv; scored with score_run49.py (orig table edited to these seeds only).

**Outcome.** Binary as in run43: early = some L1 head with j0+2 ≥ 0.50 and j0+1 ≤ 0.10, else not-early. Kept = same binary route as run43. The 3-way label (early / classic / other) is also reported.

**Primary.** ≥ 10/12 kept → init decides even for borderline inits. ≤ 8/12 → data order tips borderline inits. 9 → inconclusive.

**Secondary.** Marker AUC (early vs not) over the 12 runs; marker ranges per init. Reported with the same pseudo-replication caveat (4 inits).

**Forecasts (logged now).**
- P(≥ 10/12 kept) = 40%.
- P(≤ 8/12 kept) = 35%.
- P(at least one of the 6 s32/s33 runs comes out early) = 50%.
- P(marker AUC ≥ 0.70 | both classes ≥ 2) = 55%.
