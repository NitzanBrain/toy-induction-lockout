# Score run49 (PREREG-run49.md): route per run (frozen L1 rule), marker (mean over evals 1000-1500 of max_h(early-classic)), escape step.
import re, sys, itertools
txt = open(sys.argv[1]).read().split("== ")[1:]
orig = {47: "early", 36: "early", 39: "classic", 45: "classic"}
rows = []
for blk in txt:
    s, d = map(int, re.search(r"init seed (\d+), DATA_SEED (\d+)", blk).groups())
    mk, step, esc = [], None, None
    for line in blk.splitlines():
        m = re.match(r"step\s+(\d+) .*loss on copies ([\d.]+)", line)
        if m:
            step = int(m.group(1)); 
            if esc is None and float(m.group(2)) < 1.0: esc = step
        m = re.search(r"L1 early \[(.*?)\] \| L1 classic \[(.*?)\]", line)
        if m and 1000 <= step <= 1500:
            e = list(map(float, m.group(1).split(","))); c = list(map(float, m.group(2).split(",")))
            mk.append(max(a - b for a, b in zip(e, c)))
    heads = [tuple(map(float, x)) for x in re.findall(r"j0\+1 \(classic target\) ([\d.]+) \| on j0\+2 ([\d.]+)", blk)]
    if any(j2 >= 0.5 and j1 <= 0.1 for j1, j2 in heads): r = "early"
    elif any(j1 >= 0.5 and j2 <= 0.1 for j1, j2 in heads): r = "classic"
    else: r = "other"
    rows.append((s, d, r, sum(mk) / len(mk), len(mk), esc))
for s, d, r, m, n, e in rows:
    print(f"s{s} d{d}: route {r:8s} orig {orig[s]:8s} kept {r == orig[s]} | marker {m:.4f} (n={n}) | escape {e}")
kept = sum(r == orig[s] for s, _, r, *_ in rows); other = sum(r == "other" for _, _, r, *_ in rows)
print(f"kept {kept}/12; other {other}; kept among non-other {kept}/{12 - other}")
pos = [m for _, _, r, m, *_ in rows if r == "early"]; neg = [m for _, _, r, m, *_ in rows if r != "early"]
U = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg); auc = U / (len(pos) * len(neg))
allm = [m for *_, m, n, e in [(x[0], x[1], x[2], x[3], x[4], x[5]) for x in rows]]
k = len(pos); ge = tot = 0
for comb in itertools.combinations(range(12), k):
    pp = [allm[i] for i in comb]; nn = [allm[i] for i in range(12) if i not in comb]
    u = sum((p > q) + 0.5 * (p == q) for p in pp for q in nn); tot += 1; ge += u >= U - 1e-9
print(f"marker AUC early vs not: {auc:.3f} (U={U}/{len(pos)*len(neg)}), exact one-sided p = {ge}/{tot} = {ge/tot:.4f}")
for s in orig:
    ms = [m for ss, _, _, m, *_ in rows if ss == s]; print(f"  s{s} markers {[round(x,4) for x in ms]} range {max(ms)-min(ms):.4f}")
