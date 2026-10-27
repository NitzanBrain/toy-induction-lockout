# Which route feeds each induction head? For every induction head (score on the classic target > 0.5) and every head in
# an earlier layer, zero that earlier head in the residual stream that feeds ONLY the induction head's queries, or ONLY
# its keys (everything between is recomputed), and measure the induction head's attention on its target.
# Generalises analyze_qk.py (which did one L0 -> L1 pair). Evaluated on the training distribution if BIGRAM is set.
# Run: [THRESH=0.5] <venv>/bin/python -I analyze_route.py <model.pt>
import sys
src = open(__file__.replace("analyze_route.py", "analyze_copy.py")).read().split("base = copy_loss()")[0]
exec(src)
tri = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
same = (x[:, :, None] == x[:, None, :]); prevm = torch.zeros_like(same); prevm[:, :, 1:] = same[:, :, :-1]
causal = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
def stream_before(layer, kill=None):
    """Residual stream entering `layer`; kill = (l, h) zeroes head h of layer l on the way."""
    with torch.no_grad():
        h = m.emb(x) + m.pos(torch.arange(T))
        for k in range(layer):
            L = m.layers[k]
            if kill and kill[0] == k:
                mk = torch.ones(H); mk[kill[1]] = 0; L.mask = mk
            h = h + L(h); L.mask = None
            if MLPF: h = h + m.mlps[k](h)
        return h
def target_mass(layer, head, hq, hk):
    with torch.no_grad():
        L = m.layers[layer]
        q = L.qkv(hq).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4)[0][:, head]
        k = L.qkv(hk).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4)[1][:, head]
        A = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(tri, -1e9).softmax(-1)
        return (A * (prevm & causal)).sum(-1)[cm].mean().item()
for li in range(1, NL):
    clean = stream_before(li)
    for hh in range(H):
        base = target_mass(li, hh, clean, clean)
        if base < float(os.environ.get("THRESH", 0.5)): continue
        print(f"induction head L{li}H{hh}: clean target mass {base:.2f}")
        rows = []
        for lj in range(li):
            for hj in range(H):
                abl = stream_before(li, (lj, hj))
                rows.append((lj, hj, target_mass(li, hh, abl, clean), target_mass(li, hh, clean, abl)))
        rows.sort(key=lambda r: min(r[2], r[3]))
        for lj, hj, mq, mk in rows[:4]:
            print(f"   remove L{lj}H{hj}: from Q -> {mq:.2f} | from K -> {mk:.2f}")
