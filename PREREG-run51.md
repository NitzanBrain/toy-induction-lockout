# run51: fresh-seed test of the MEAN-over-heads layer-1 marker (pre-registered 2026-11-02 ~10:20, before any training of seeds 48–63)

**Origin:** explore-marker2 (RESULTS.md), exploratory: mean over L1 heads of (early − classic) beat the run43 max marker slightly (run43 0.867 vs 0.85; run49+50 pooled 0.815 vs 0.756), after trying 5 variants.

**Seeds:** 48–63 (16 seeds, never trained). One run each, default data order (no DATA_SEED).
**Flags:** HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5, `train.py 4 8000 <seed>`, SAVE, then `analyze_qkv.py <pt> 0`.

**Markers (both frozen, from HEADLOG2 lines at evals in steps 1000–1500):**
- MEAN: average over evals of the mean over the 4 L1 heads of (L1 early − L1 classic). (primary)
- MAX: average over evals of the max over heads of the same difference (run43 marker). (comparison)

**Outcome (frozen):** one-step-early if some L1 head has j0+2 ≥ 0.50 and j0+1 ≤ 0.10 at step 8000; else not-early. 3-way label also reported. Runs with copy loss never < 1.0 are excluded and reported.

**Criteria:**
- Primary: AUC(MEAN) ≥ 0.70, exact one-sided permutation p reported. Inconclusive if fewer than 2 early seeds.
- Secondary: AUC(MEAN) > AUC(MAX). Reported as a difference only; with 16 seeds it can't be significant, so no claim of "better" unless the difference is ≥ 0.10.

**Forecasts (logged now):** P(≥ 2 early) = 75%. P(AUC(MEAN) ≥ 0.70 | ≥ 2 early) = 55%. P(AUC(MEAN) > AUC(MAX)) = 55%. P(difference ≥ 0.10) = 20%.
