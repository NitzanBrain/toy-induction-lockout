# Held-out route prediction (run35): predictor = max over L0 heads of the prev-token score, averaged over evals in
# steps [500, 900]. AUC of "K seeds score higher" over the 14 seeds not used to form the hypothesis (run34 used s5, s7, s12, s13).
# Run: python3 -I early_predict.py run35-early.txt
import re, ast, sys
Q = {1, 4, 7, 11, 13, 15}; D = {}; cur = None; st = None
for l in open(sys.argv[1]):
    m = re.match(r'== RANDSTART L=4 seed (\d+)', l)
    if m: cur = int(m.group(1)); D[cur] = []; continue
    m = re.match(r'step +(\d+) ', l)
    if m: st = int(m.group(1)); continue
    m = re.match(r'  HEADS L0 prev (\[.*?\])', l)
    if m and 500 <= st <= 900: D[cur].append(max(ast.literal_eval(m.group(1))))
sc = {s: sum(v) / len(v) for s, v in D.items()}
for s in sorted(sc): print(f"s{s} {'Q' if s in Q else 'K'} {sc[s]:.3f}")
q = [sc[s] for s in sc if s in Q]; k = [sc[s] for s in sc if s not in Q]
auc = sum((b > a) + 0.5 * (b == a) for a in q for b in k) / (len(q) * len(k))
print(f"AUC (K higher) = {auc:.3f}  n_Q={len(q)} n_K={len(k)}")
