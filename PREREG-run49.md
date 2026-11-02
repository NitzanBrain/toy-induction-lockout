# run49: does the initialization decide the layer-1 route, or the data order? (pre-registered 2026-11-02 ~05:00, before any run49 training)

**Question.** run43 showed that a small early layer-1 lean (the marker) predicts the route, and run44–48 showed that pushes of about the marker's size can flip it. Still open: in unpushed runs, is the route fixed by the initial weights, or does the stochastic batch order decide it? The marker is measured after 1000 steps of data, so it could reflect either.

**Change to train.py (only this).** If env DATA_SEED is set, `torch.manual_seed(DATA_SEED)` is called after the model, optimiser and eval set are built. Init and eval set are those of `seed`; only the training batches change. Without DATA_SEED nothing changes.

**Runs (12, no bias).**
- Init seeds: s47, s36 (one-step-early in run43; markers ranked 1st and 2nd); s39, s45 (classic in run43 and in the run46 unbiased baseline).
- Data seeds: 1001, 1002, 1003 for each init.
- Flags: HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5, `train.py 4 8000 <seed>`, SAVE, then `analyze_qkv.py <pt> 0`.

**Outcome (frozen, as run44–48).** One-step-early if some L1 head has j0+2 ≥ 0.50 and j0+1 ≤ 0.10; classic if some L1 head has j0+1 ≥ 0.50 and j0+2 ≤ 0.10 and no head meets the early rule; otherwise "other" (reported). Runs with copy loss never < 1.0 by 8000 are excluded and reported.

**Marker (frozen, as run43).** Mean over evals in steps 1000–1500 of max over L1 heads of (L1 early − L1 classic), from the HEADLOG2 lines of each run.

**Primary.** Route kept = the run's route equals its init's run43 route.
- "Init decides" if ≥ 10/12 kept.
- "Data order matters" if ≤ 8/12 kept.
- 9/12: inconclusive.

**Secondary.** Across the 12 runs, AUC of the marker for one-step-early vs not, if each class has ≥ 2 runs; exact one-sided permutation p. Also: per init, does the marker move with data order (range within init vs between inits)?

**Caveat stated in advance.** run43 routes were scored at 12000 steps, these at 8000. run46 used 8000 for the classic baseline. Early seeds escaped by 6300 in run43, so 8000 should be enough, but a run classified "other" at 8000 is reported as such, not counted as a flip.

**Forecasts (logged now).**
- P(≥ 10/12 kept) = 35%.
- P(≤ 8/12 kept) = 40%.
- P(marker AUC ≥ 0.70 | both classes ≥ 2) = 60%.
- Expected early→classic flips more often than classic→early (base rate of early ≈ 12/46): P(more flips among s47/s36 runs than among s39/s45 runs) = 60%.
