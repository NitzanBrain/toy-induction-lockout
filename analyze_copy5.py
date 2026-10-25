# Hypothesis: L0 heads copy the token at ≈k/2 into position k; L1 matches its query (current token x[i]) against that
# imported token, i.e. attends to keys k with x[k//2]==x[i] (or x[(k+1)//2]), and reads out the neighbour x[k/2 + 1] = the next token.
import sys, torch
sys.argv = [sys.argv[0], sys.argv[1]]; exec(open("analyze_copy.py").read().split("base = copy_loss()")[0])
with torch.no_grad(): m(x)
idx = torch.arange(T)
def match(off):  # M[r,i,k] = x[(k+off)//2] == x[i]
    src = torch.clamp((idx + off) // 2, 0, T - 1); return x[:, None, :].expand(N, T, T).gather(2, src.view(1, 1, T).expand(N, T, T)) == x[:, :, None]
causal = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
for off in [0, 1, -1, 2]:
    M = match(off) & causal
    for h in range(H):
        A = m.layers[1].pat[:, h]; mass = (A * M).sum(-1)[cm].mean().item(); base = (M.float().sum(-1) / causal.sum(-1).clamp(min=1))[cm].mean().item()
        print(f"L1H{h}: mass on keys with x[(k+{off})//2]==x[i]: {mass:.2f}   (uniform-attention expectation {base:.2f})")
