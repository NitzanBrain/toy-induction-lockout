# Score run51 (PREREG-run51.md): MEAN (primary) and MAX markers over evals 1000-1500; binary early rule at 8000; exact one-sided permutation p.
import re, sys, itertools
rows = []
for blk in open(sys.argv[1]).read().split("== ")[1:]:
    s = int(re.search(r"init seed (\d+)", blk).group(1)); step = None; D = []; esc = None
    for line in blk.splitlines():
        m = re.match(r"step\s+(\d+) .*loss on copies ([\d.]+)", line)
        if m:
            step = int(m.group(1))
            if esc is None and float(m.group(2)) < 1.0: esc = step
        m = re.search(r"L1 early \[(.*?)\] \| L1 classic \[(.*?)\]", line)
        if m and step is not None and 1000 <= step <= 1500:
            e = list(map(float, m.group(1).split(","))); c = list(map(float, m.group(2).split(","))); D.append([a - b for a, b in zip(e, c)])
    hv = [(float(a), float(b)) for a, b in re.findall(r"j0\+1 \(classic target\) ([\d.]+) \| on j0\+2 ([\d.]+)", blk)]
    early = any(j2 >= .5 and j1 <= .1 for j1, j2 in hv); classic = any(j1 >= .5 and j2 <= .1 for j1, j2 in hv)
    lab3 = "early" if early else "classic" if classic else "other"
    n = len(D); mean = sum(sum(d) / len(d) for d in D) / n; mx = sum(max(d) for d in D) / n
    rows.append((s, early, lab3, mean, mx, n, esc))
for s, e, l3, mean, mx, n, esc in rows: print(f"s{s}: {l3:7s} MEAN {mean:.4f} MAX {mx:.4f} (n={n}) escape {esc}")
excl = [r for r in rows if r[6] is None]; print("excluded (no escape):", [r[0] for r in excl])
rows = [r for r in rows if r[6] is not None]
def auc(i, R):
    P = [r[i] for r in R if r[1]]; N = [r[i] for r in R if not r[1]]
    return sum((p > q) + .5 * (p == q) for p in P for q in N) / (len(P) * len(N))
k = sum(r[1] for r in rows); print(f"early {k}/{len(rows)}")
if k >= 2:
    for i, nm in [(3, "MEAN"), (4, "MAX")]:
        A = auc(i, rows); vals = [r[i] for r in rows]; ge = tot = 0
        for comb in itertools.combinations(range(len(rows)), k):
            P = [vals[j] for j in comb]; N = [vals[j] for j in range(len(rows)) if j not in comb]
            a = sum((p > q) + .5 * (p == q) for p in P for q in N) / (len(P) * len(N)); tot += 1; ge += a >= A - 1e-12
        print(f"AUC({nm}) {A:.3f}  exact one-sided p {ge}/{tot} = {ge/tot:.4f}")
