# EXPLORATORY (not pre-registered): alternative layer-1 markers on existing logs, to separate "one head leans" from "layer commits".
# For each run: per-eval per-head d_h = early_h - classic_h over steps 1000-1500. Markers: max_h (run43 marker), mean_h, min_h, count of heads with mean d_h > 0.
import re, sys
def runs(path, hdr):
    for blk in open(path).read().split("== ")[1:]:
        name = re.match(hdr, blk); 
        if not name: continue
        step = None; D = []
        for line in blk.splitlines():
            m = re.match(r"step\s+(\d+)", line)
            if m: step = int(m.group(1))
            m = re.search(r"L1 early \[(.*?)\] \| L1 classic \[(.*?)\]", line)
            if m and step is not None and 1000 <= step <= 1500:
                e = list(map(float, m.group(1).split(","))); c = list(map(float, m.group(2).split(",")))
                D.append([a - b for a, b in zip(e, c)])
        heads = [x for x in re.findall(r"j0\+1 \(classic target\) ([\d.]+) \| on j0\+2 ([\d.]+)", blk)]
        hv = [(float(a), float(b)) for a, b in heads]
        lab = "early" if any(j2 >= .5 and j1 <= .1 for j1, j2 in hv) else "not"
        yield name.group(0).strip(), D, lab
def auc(pos, neg): return sum((p > q) + .5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
rows = []
for path in sys.argv[1:]:
    for nm, D, lab in runs(path, r"[^\n]*seed \d+[^\n]*"):
        if not D: continue
        n = len(D); H = len(D[0])
        mh = [sum(d[h] for d in D) / n for h in range(H)]
        rows.append((nm[:60], lab, sum(max(d) for d in D) / n, sum(mh) / H, min(mh), sum(x > 0 for x in mh), sorted(mh)[-2]))
names = ["max (run43)", "mean over heads", "min head", "#heads >0", "2nd-highest head"]
for i, nmk in enumerate(names):
    pos = [r[2 + i] for r in rows if r[1] == "early"]; neg = [r[2 + i] for r in rows if r[1] != "early"]
    print(f"{nmk:18s} AUC {auc(pos, neg):.3f}  (early n={len(pos)}, not n={len(neg)})")
for r in rows: print(r[1], " ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r[2:]), "|", r[0])
