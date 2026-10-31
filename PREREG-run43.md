# run43: prospective test of an early marker for the ONE-STEP-EARLY route (pre-registered 2026-10-30 ~21:20, before any training of seeds 32–47)

**Origin:** run42b, exploratory, 30 seeds. The marker separated one-step-early from classic seeds with AUC 0.85 in window 1000–1500, after forking paths: 6 measures × 4 windows. This file tests it once on fresh seeds.

**New seeds:** 32–47 (16 seeds, never trained).

**Marker (frozen):**
- For each eval step in [1000, 1500]: max over the 4 layer-1 heads of (L1 early − L1 classic), from the HEADLOG2 line "L1 early [...] | L1 classic [...]".
- Average over those evals.
- Source: HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5 train.py 4 1500 <seed>.

**Outcome (frozen):**
- Train 12000 steps with the same flags and SAVE.
- Run analyze_qkv.py on the saved model (layer0_head argument = 0; the L1 mass lines don't depend on it).
- A seed is **one-step-early** if some layer-1 head has mass on j0+2 ≥ 0.50 AND mass on j0+1 ≤ 0.10. Otherwise it is **classic**.
- Seeds that never reach copy loss < 1.0 by 12000 are excluded and reported.

**Criteria:**
- Primary: AUC (one-step-early seeds score higher) ≥ 0.70.
- If fewer than 2 one-step-early seeds appear → inconclusive. Expected count ≈ 3 (base rate 6/30).
- Also reported: exact one-sided permutation p over all splits. No cut-off rule is tested.

**My forecast (logged now):** P(AUC ≥ 0.70 | ≥2 positives) = 45%; P(≥2 positives) = 70%.
