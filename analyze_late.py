# Random-start 4L s0 (run28d): which later layers carry layer 1's answer forward? (1) zero whole L2/L3 attention, and
# L2+L3 except one head; (2) cleaner path test: later-layer attention VALUES computed from a stream without layer 1.
# Run: RANDSTART=1 BIGRAM=0.5 <venv>/bin/python -I analyze_late.py <model.pt>
import sys
src = open(__file__.replace("analyze_late.py", "analyze_copy.py")).read().split("base = copy_loss()")[0]
exec(src)
base = copy_loss(); print(f"copy loss clean {base:.3f}")
for name, layers in (("L2 attention zeroed", [2]), ("L3 attention zeroed", [3]), ("L2 and L3 attention zeroed", [2, 3])):
    for li in layers: m.layers[li].mask = torch.zeros(H)
    print(f"{name}: {copy_loss():.3f}")
    for li in layers: m.layers[li].mask = None
# all heads of L2+L3 except one: which single head suffices?
for keep in [(2, h) for h in range(H)] + [(3, h) for h in range(H)]:
    for li in (2, 3):
        mk = torch.zeros(H)
        if keep[0] == li: mk[keep[1]] = 1
        m.layers[li].mask = mk
    print(f"L2+L3 zeroed except L{keep[0]}H{keep[1]}: {copy_loss():.3f}")
    for li in (2, 3): m.layers[li].mask = None
# Cleaner path test: does layer 1's output reach the prediction THROUGH later attention VALUES (carried from i−1 to i)?
# Build a parallel stream with layer 1's attention output removed (recomputed through MLP1 and later layers). In the real
# stream, compute each later layer's attention VALUES from the parallel stream (Q, K from the real stream), or the reverse.
tri = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
def attn(L, hq, hv):
    q, k, _ = L.qkv(hq).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
    v = L.qkv(hv).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4)[2]
    a = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(tri, -1e9).softmax(-1)
    return L.o((a @ v).transpose(1, 2).reshape(N, T, D))
def loss_of(h):
    ce = F.cross_entropy(m.un(h)[:, :-1].transpose(1, 2), x[:, 1:], reduction="none"); return ce[cm[:, :-1]].mean().item()
with torch.no_grad():
    h0 = m.emb(x) + m.pos(torch.arange(T)); h = h0 + m.layers[0](h0); h = h + m.mlps[0](h)
    real = h + m.layers[1](h); real = real + m.mlps[1](real)
    para = h.clone(); para = para + m.mlps[1](para)          # layer-1 attention output removed
    for mode in ("clean (sanity)", "later-layer VALUES without layer 1", "layer 1 removed everywhere"):
        r, p_ = real.clone(), para.clone()
        if mode == "layer 1 removed everywhere": r = para.clone()
        for kk in range(2, NL):
            L = m.layers[kk]
            if mode == "later-layer VALUES without layer 1": r = r + attn(L, r, p_)
            else: r = r + attn(L, r, r)
            p_ = p_ + attn(L, p_, p_)
            r = r + m.mlps[kk](r); p_ = p_ + m.mlps[kk](p_)
        print(f"{mode}: copy loss {loss_of(r):.3f}")
# Per-layer version (review 10-27): parallel-stream values in ONE later layer only (the other later layer fully clean).
with torch.no_grad():
    for only in range(2, NL):
        r, p_ = real.clone(), para.clone()
        for kk in range(2, NL):
            L = m.layers[kk]
            r = r + (attn(L, r, p_) if kk == only else attn(L, r, r))
            p_ = p_ + attn(L, p_, p_)
            r = r + m.mlps[kk](r); p_ = p_ + m.mlps[kk](p_)
        print(f"values without layer 1 in layer {only} only: copy loss {loss_of(r):.3f}")
