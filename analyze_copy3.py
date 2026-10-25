# Is layer 0's "look ~0.8 s back" positional (k ≈ i/2) or content-based (k ≈ i−s+1)? Compare argmax key with both, and test
# on sequences WITHOUT any repeat (no copy at all): if the head still looks at ~i/2, it is positional.
import sys, torch
sys.argv = [sys.argv[0], sys.argv[1]]; exec(open("analyze_copy.py").read().split("base = copy_loss()")[0])
def stats(xx, SS, rep):
    with torch.no_grad(): m(xx)
    for h in range(3):
        A = m.layers[0].pat[:, h]; e_half = e_tgt = n = 0
        for r in range(0, N, 2):
            s = SS[r].item()
            for i in range(s, 2 * s - 1):
                k = int(A[r, i].argmax()); e_half += abs(k - i / 2); e_tgt += abs(k - (i - s + 1)); n += 1
        print(f"  {rep} L0H{h}: mean |k − i/2| {e_half/n:.2f}   mean |k − (i−s+1)| {e_tgt/n:.2f}")
stats(x, S, "with copy   ")
x2 = torch.randint(0, V, (N, T)); stats(x2, S, "no copy     ")
