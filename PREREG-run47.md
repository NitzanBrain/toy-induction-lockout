# run47: smaller dose (0.05 → j0+2) + reverse small dose (0.1 → j0+1). Pre-registered 2026-11-01 ~05:20, before training

**Motivation:** an exploratory, back-of-envelope computation from run42b HEADLOG2. Done BEFORE this file; it is not part of the test.
- Mean L1 attention on key j0+2 at steps 1000–1500 is 0.0234.
- A logit bias b shifts it by about p(e^b − 1)/(1 + p(e^b − 1)):
  - 0.05 → 0.0012
  - 0.1 → 0.0025
  - 0.25 → 0.0066
  - 0.5 → 0.0150
- The natural marker (max_h early − classic, window 1000–1500) ranges:
  - classic seeds 0.0006–0.0056;
  - one-step-early seeds 0.0025–0.0064.
- So the 0.1 push that flipped 2/3 in run46 is of the same order as natural variation.
- Caveat: the bias hits every head and every copy position for 2500 steps, while the marker is one head's average preference.

**Runs:** RANDSTART=1 MLP=1 BIGRAM=0.5, 4 layers, 8000 steps, BIASL1_STEPS=2500. Same code as run44–46.
- **(A)** BIASL1=0.05, OFF=2 (toward j0+2) on classic seeds s39, s45, s40.
- **(B)** BIASL1=0.1, OFF=1 (toward j0+1) on one-step-early seeds s47, s36, s44.

**Outcome:** analyze_qkv L1 masses at step 8000.
- **one-step-early:** some L1 head has j0+2 ≥ 0.50 AND j0+1 ≤ 0.10 (the run43 rule).
- **classic (for B):** NOT one-step-early AND some L1 head has j0+1 ≥ 0.50 AND j0+2 ≤ 0.10.
- **mixed:** anything else. Reported as such.

**Predictions:**
- (A) ≥ 2/3 one-step-early. P = 35% (0.1 gave 2/3; 0.05 is half the shift).
- (B) ≥ 2/3 classic. P = 65% (0.5 gave 3/3 with escape at 1800; classic is the majority route, 24/30).

**Interpretation fixed now:**
- If (A) holds at 0.05, the flip threshold for 2 of 3 seeds is at or below ~0.001 attention, which is inside the natural marker range.
- If (B) holds, the dose sensitivity is symmetric at 0.1.
- n = 3 per arm; no p-values; reported as counts.
