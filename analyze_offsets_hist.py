# Where do the layer-0/1 heads look? Histogram of i − argmax attention over all positions (run28b).
# Run: [RANDSTART=1] BIGRAM=0.5 <venv>/bin/python -I analyze_offsets_hist.py <model.pt>
import sys, collections
src = open(__file__.replace("analyze_offsets_hist.py", "analyze_copy.py")).read().split("base = copy_loss()")[0]
exec(src)
with torch.no_grad(): m(x)
for li in (0, 1):
    for h in range(H):
        A = m.layers[li].pat[:, h]
        c = collections.Counter()
        for r in range(0, N, 4):
            for i in range(4, T):
                c[i - int(A[r, i].argmax())] += 1
        tot = sum(c.values())
        print(f"L{li}H{h} all positions, i - argmax: " + ", ".join(f"{d}:{v / tot:.2f}" for d, v in c.most_common(5)))
