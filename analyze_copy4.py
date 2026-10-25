# Composition test: does layer 1 read layer 0's output through its Query, Key or Value? Recompute L1 with Q/K/V inputs taken
# either from h1 = emb+pos+L0(·) (normal) or from h0 = emb+pos (L0 removed for that path only); final residual keeps L0's direct output.
import sys, math, itertools, torch, torch.nn.functional as F
sys.argv = [sys.argv[0], sys.argv[1]]; exec(open("analyze_copy.py").read().split("base = copy_loss()")[0])
L0, L1 = m.layers
with torch.no_grad():
    h0 = m.emb(x) + m.pos(torch.arange(T)); o0 = L0(h0); h1 = h0 + o0
    Wq, Wk, Wv = L1.qkv.weight[:D], L1.qkv.weight[D:2*D], L1.qkv.weight[2*D:]
    def run(qi, ki, vi):
        b, t = x.shape; sh = lambda z: z.view(b, t, H, D // H).transpose(1, 2)
        q, k, v = sh(qi @ Wq.T), sh(ki @ Wk.T), sh(vi @ Wv.T)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), -1e9).softmax(-1)
        o1 = L1.o((a @ v).transpose(1, 2).reshape(b, t, D)); lg = m.un(h1 + o1)
        ce = F.cross_entropy(lg[:, :-1].transpose(1, 2), x[:, 1:], reduction="none"); return ce[cm[:, :-1]].mean().item()
    for qn, kn, vn in itertools.product(["h1", "h0"], repeat=3):
        d = {"h1": h1, "h0": h0}; print(f"Q from {qn}, K from {kn}, V from {vn}: copy loss {run(d[qn], d[kn], d[vn]):.3f}")
