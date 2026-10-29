# Pooled escape-timing test (run32 + run33): do Q-route seeds escape later than K-route seeds? Escape step = first eval
# with copy loss < 1.0. Exact one-sided permutation test of the Mann-Whitney U (ties count 0.5) over all labelings.
# Run: python3 -I timing_test.py run33-timing.txt run32-seeds-escape.txt
import re, sys, itertools
esc = {}
for f in sys.argv[1:]:
    cur = None
    for l in open(f):
        m = re.match(r'== RANDSTART L=4 seed (\d+)', l)
        if m: cur = int(m.group(1)); continue
        m = re.match(r'step +(\d+) .*copies ([\d.]+)', l)
        if m and cur is not None and cur not in esc and float(m.group(2)) < 1.0: esc[cur] = int(m.group(1))
Q = {1, 4, 7, 11, 13, 15}   # layer-1 route from run27-32 (s10 counted K: its strong head)
print("escape steps:", dict(sorted(esc.items())))
seeds = sorted(esc); vals = [esc[s] for s in seeds]
def U(qi): return sum((vals[a] > vals[b]) + 0.5 * (vals[a] == vals[b]) for a in qi for b in range(len(vals)) if b not in qi)
qi = [i for i, s in enumerate(seeds) if s in Q]; u = U(qi)
null = [U(c) for c in itertools.combinations(range(len(vals)), len(qi))]
qs = sorted(esc[s] for s in seeds if s in Q); ks = sorted(esc[s] for s in seeds if s not in Q)
print(f"Q {qs} median {qs[len(qs)//2-1:len(qs)//2+1]} | K {ks}")
print(f"U = {u} of {len(qi) * (len(vals) - len(qi))}, one-sided p = {sum(x >= u for x in null) / len(null):.4f} (n_Q={len(qi)}, n_K={len(vals)-len(qi)})")
missing = [s for s in range(18) if s not in esc]; print("never escaped within budget:", missing)
