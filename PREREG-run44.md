# run44: causal test of the one-step-early marker (pre-registered 2026-10-31 ~00:55, before any run44 training)

**Question:** run43 showed that an early L1 preference for key j0+2 over j0+1 predicts the one-step-early circuit (AUC 0.85, p = 0.011). Does nudging it change the circuit?

**Interventions:** train.py `BIASL1=0.5 BIASL1_STEPS=2500`, layer 1 only, on copy positions only. All other flags as run43 (RANDSTART=1 MLP=1 BIGRAM=0.5), 8000 steps.
- **A. Toward one step early** (`BIASL1_OFF=2`, key j0+2) on 3 classic seeds from run43: s39, s45, s40.
- **B. Toward classic** (`BIASL1_OFF=1`, key j0+1) on 3 one-step-early seeds from run43: s47, s36, s44.

**Outcome (frozen):** same rule as run43, on the step-8000 model via analyze_qkv.py. One-step-early = some L1 head with j0+2 ≥ 0.50 and j0+1 ≤ 0.10.

**Predictions (logged now):**
- A: ≥ 2/3 become one-step-early. P = 55% (the push may simply be undone after step 2500, or route the circuit elsewhere).
- B: ≥ 2/3 become classic. P = 70% (pushing the classic target is likely to help a classic head form).

**Caveat stated in advance:** +0.5 logit on a specific key is a strong, targeted push. A flip shows the early preference can steer the circuit. It does not show that the natural ~0.005 attention difference is what steers it.
