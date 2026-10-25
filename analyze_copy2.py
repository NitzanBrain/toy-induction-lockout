# Where do the low-entropy heads put their mass? Argmax-key histograms on copy positions: absolute key position and i−k.
import sys, collections, torch
sys.argv = [sys.argv[0], sys.argv[1]]; exec(open("analyze_copy.py").read().split("base = copy_loss()")[0])
with torch.no_grad(): m(x)
for li, l in enumerate(m.layers):
    for h in range(H):
        ca, cd, cr = collections.Counter(), collections.Counter(), collections.Counter()
        for r in range(0, N, 2):
            s = S[r].item()
            for i in range(s, 2 * s - 1):
                k = int(l.pat[r, h, i].argmax()); ca[k] += 1; cd[i - k] += 1; cr[round((i - k) / s, 1)] += 1
        tot = sum(ca.values())
        print(f"L{li}H{h}: abs key {[(k, round(v/tot,2)) for k, v in ca.most_common(3)]} | i−k {[(k, round(v/tot,2)) for k, v in cd.most_common(3)]} | (i−k)/s {[(k, round(v/tot,2)) for k, v in cr.most_common(3)]}")
