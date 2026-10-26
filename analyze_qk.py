# Path patching for the 4-layer seed-1 model (run22): does the induction head L1H0 read the duplicate-token head L0H1
# through its QUERY or its KEY? We rebuild the layer-1 input twice: with L0H1 present (clean) and with L0H1 zeroed
# (MLP0 recomputed on the ablated stream). Then L1H0's queries come from one stream and its keys from the other.
# Measured on copy positions: L1H0's attention mass on the induction target (x[j-1]==x[i]) and the model's copy loss when
# only L1H0's pattern is patched.
# Run: <venv>/bin/python -I analyze_qk.py <model.pt> [layer0_head] [layer1_head]
import sys
src = open(__file__.replace("analyze_qk.py", "analyze_copy.py")).read().split("base = copy_loss()")[0]
H0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1; H1 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.argv = sys.argv[:2]; exec(src)
L0, L1 = m.layers[0], m.layers[1]
with torch.no_grad():
    h0 = m.emb(x) + m.pos(torch.arange(T))
    def after_layer0(mask):
        L0.mask = mask; h = h0 + L0(h0); L0.mask = None
        return h + m.mlps[0](h) if MLPF else h
    clean = after_layer0(None)
    mk = torch.ones(H); mk[H0] = 0; abl = after_layer0(mk)
    def qkv(h):
        q, k, v = L1.qkv(h).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4); return q, k, v
    qc, kc, vc = qkv(clean); qa, ka, _ = qkv(abl)
    tri = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    def pattern(q, k):
        return ((q[:, H1] @ k[:, H1].transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(tri, -1e9).softmax(-1)
    same = (x[:, :, None] == x[:, None, :]); prevm = torch.zeros_like(same); prevm[:, :, 1:] = same[:, :, :-1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
    def tgt_mass(A): return (A * (prevm & causal)).sum(-1)[cm].mean().item()
    def loss_with(A):
        # rerun the model from layer 1 with L1H{H1}'s pattern replaced by A (values from the clean stream)
        h = clean
        b, t, _ = h.shape
        q, k, v = L1.qkv(h).view(b, t, 3, H, D // H).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(tri, -1e9).softmax(-1)
        a[:, H1] = A; z = a @ v; h = h + L1.o(z.transpose(1, 2).reshape(b, t, D))
        if MLPF: h = h + m.mlps[1](h)
        for kk in range(2, NL):
            h = h + m.layers[kk](h)
            if MLPF: h = h + m.mlps[kk](h)
        lg = m.un(h); ce = F.cross_entropy(lg[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
        return ce[cm[:, :-1]].mean().item()
    for name, (q, k) in {"clean Q, clean K": (qc, kc), "ablated Q, clean K": (qa, kc),
                         "clean Q, ablated K": (qc, ka), "ablated Q, ablated K": (qa, ka)}.items():
        A = pattern(q, k)
        print(f"L0H{H0} -> L1H{H1} | {name:22s}: target mass {tgt_mass(A):.2f} | copy loss with this pattern {loss_with(A):.3f}")
