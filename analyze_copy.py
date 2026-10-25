# Mechanism of the "non-classic copying" after an abrupt shortcut removal (run15/16). Loads a model saved by train.py SAVE=...
# 1) per-head zero-ablation: copy loss with one head's output removed; 2) where attention goes on copy positions, by CONTENT:
# mass on keys j with x[j]==x[i] (same token), x[j-1]==x[i] (classic induction target), x[j+1]==x[i]; and by offset d−(s−1).
# Run: <venv>/bin/python -I analyze_copy.py <model.pt>
import sys, math, torch, torch.nn as nn, torch.nn.functional as F
ck = torch.load(sys.argv[1], weights_only=False); NL, H, MLPF = ck["NL"], ck["H"], ck["MLP"]; V, T, D = 64, 64, 64
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.qkv = nn.Linear(D, 3 * D, bias=False); s.o = nn.Linear(D, D, bias=False); s.mask = None
    def forward(s, x):
        b, t, _ = x.shape; q, k, v = s.qkv(x).view(b, t, 3, H, D // H).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), -1e9).softmax(-1)
        s.pat = a; z = a @ v
        if s.mask is not None: z = z * s.mask.view(1, H, 1, 1)
        return s.o(z.transpose(1, 2).reshape(b, t, D))
class Model(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(T, D); s.layers = nn.ModuleList(Attn() for _ in range(NL)); s.un = nn.Linear(D, V, bias=False)
        s.mlps = nn.ModuleList(nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) for _ in range(NL))
    def forward(s, x):
        h = s.emb(x) + s.pos(torch.arange(x.shape[1]))
        for k, l in enumerate(s.layers):
            h = h + l(h)
            if MLPF: h = h + s.mlps[k](h)
        return s.un(h)
m = Model(); m.load_state_dict(ck["state"]); m.eval()
torch.manual_seed(99); N = 512; x = torch.randint(0, V, (N, T)); S = torch.randint(8, 33, (N,))
for r in range(N): s = S[r].item(); x[r, s:2 * s] = x[r, :s]
cm = torch.zeros(N, T, dtype=torch.bool)
for r in range(N): s = S[r].item(); cm[r, s:2 * s - 1] = True
def copy_loss():
    with torch.no_grad(): lg = m(x); ce = F.cross_entropy(lg[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
    return ce[cm[:, :-1]].mean().item()
base = copy_loss(); print(f"copy loss clean {base:.3f}")
for li, l in enumerate(m.layers):
    out = []
    for h in range(H):
        mk = torch.ones(H); mk[h] = 0; l.mask = mk; out.append(round(copy_loss() - base, 3)); l.mask = None
    l.mask = torch.zeros(H); allz = copy_loss() - base; l.mask = None
    print(f"L{li} ablate one head Δcopy loss {out} | whole layer {allz:+.3f}")
with torch.no_grad(): m(x)
same = (x[:, :, None] == x[:, None, :])                       # [r, i, j]: x[j]==x[i]
prevm = torch.zeros_like(same); prevm[:, :, 1:] = same[:, :, :-1]   # x[j-1]==x[i]
nextm = torch.zeros_like(same); nextm[:, :, :-1] = same[:, :, 1:]   # x[j+1]==x[i]
causal = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)       # strictly earlier keys
for li, l in enumerate(m.layers):
    for h in range(H):
        A = l.pat[:, h]; sel = cm
        def mass(M): return (A * (M & causal)).sum(-1)[sel].mean().item()
        offs = []
        for dd in range(-3, 4):
            v = [A[r, i, i - (S[r].item() - 1) + dd].item() if 0 <= i - (S[r].item() - 1) + dd <= i else 0.0 for r in range(0, N, 8) for i in range(S[r].item(), 2 * S[r].item() - 1)]
            offs.append(round(sum(v) / len(v), 2))
        ent = (-(A * (A + 1e-12).log()).sum(-1))[sel].mean().item()
        print(f"L{li}H{h}: mass same-token {mass(same):.2f} | x[j-1]==x[i] {mass(prevm):.2f} | x[j+1]==x[i] {mass(nextm):.2f} | self {A.diagonal(dim1=1, dim2=2)[sel].mean():.2f} | entropy {ent:.2f} | offset (s-1)+[-3..3] {offs}")
