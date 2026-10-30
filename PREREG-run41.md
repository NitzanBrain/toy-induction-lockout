# run41: prospective test of the early-lean route predictor (pre-registered 2026-10-30, before any run41 training beyond 1000 steps)

**Claim tested:** the early L0 prev-token lean predicts which route (K or Q) a 4L random-start skip-one model takes (run35: held-out AUC 0.85, n = 14).

**New seeds:** 20–31 (12 seeds; none were ever trained before; s20 got a 1000-step timing probe only).

**Predictor (frozen, identical to early_predict.py):** max over L0 heads of the prev-token score, averaged over evals in steps [500, 900].
Source: HEADLOG=1 RANDSTART=1 MLP=1 BIGRAM=0.5 train.py 4 1000 <seed>.

**Rule (frozen):** score < 0.178 → predict Q, else → predict K.
The cut comes from run35 data: it puts 3/4 Q seeds and 1/10 K seeds below it.

**Outcome (frozen, identical to summarize_run32.py):** train 12000 steps, then run analyze_route.py with THRESH=0.3. Seed route = the single biggest relative drop of target mass from removing one earlier head from Q only vs K only.
- "One-step-early" variants count by their K/Q route.
- Seeds that never escape by 12000 (copy loss ≥ 1.0) are reported separately and excluded from scoring.

**Success criteria (set now):**
- Primary: AUC (K seeds score higher) on the new seeds ≥ 0.70.
- Secondary: rule accuracy > the always-K baseline on the same seeds.
- If fewer than 2 Q seeds appear, AUC is undefined → report it as inconclusive, not as success.

**Prediction of the base rate:** about 1/3 Q (6/18 in run31/32).
