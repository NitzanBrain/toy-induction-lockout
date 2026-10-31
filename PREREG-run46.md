# run46: dose response + unbiased baseline for §5.8 (pre-registered 2026-10-31 ~03:05, before training)
**Runs:** classic seeds s39, s45, s40 (run43), 8000 steps, RANDSTART=1 MLP=1 BIGRAM=0.5.
- (a) No bias (baseline at 8000).
- (b) BIASL1=0.1, OFF=2, STEPS<2500.
- (c) BIASL1=0.25, OFF=2, STEPS<2500.
**Outcome:** run43 rule on analyze_qkv masses at step 8000.
**Predictions:**
- (a) 0/3 one step early (they were classic at 12000). P = 85%.
- (b) ≥ 2/3 one step early. P = 30%.
- (c) ≥ 2/3 one step early. P = 50%.
- Threshold somewhere in 0.1–0.5, by analogy with layer 0 (0.1–0.25).
