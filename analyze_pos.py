# Open question from article §4: in the 4L seed-1 model, L0H1 (duplicate-token head) feeds the induction head L1H0 through
# its QUERY. How do L1H0's keys single out the position right AFTER the earlier occurrence? Hypothesis: positional embeddings.
#   query at i ≈ "position of the earlier same token j0" (L0H1 copies pos(j0) through its values), key at j ≈ pos(j);
#   W_QK matches pos(j0) against pos(j0+1).
# Tests (training distribution if BIGRAM is set). The rebuilt layer-1 input is h0 + L0(h0) [+ MLP0], recomputed after edits.
#   K-side: the stream feeding only L1H0's keys is rebuilt with (a) positions permuted, (b) positions shifted (pos(j+d)),
#           (c) tokens replaced by random tokens. If keys use position, (b) should move attention by -d positions.
#   Q-side: L0H1's values computed from token embedding only or position embedding only (the stream feeding only the
#           queries); and positions in L0H1's values shifted by d, which should move attention by +d positions.
# Reported: L1H0 attention mass on the induction target (x[j-1]==x[i]), on the same token (x[j]==x[i]), and at x[j-2]==x[i].
# Run: <venv>/bin/python -I analyze_pos.py <model.pt> [layer0_head] [layer1_head]
import sys
src = open(__file__.replace("analyze_pos.py", "analyze_copy.py")).read().split("base = copy_loss()")[0]
H0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1; H1 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.argv = sys.argv[:2]; exec(src)
L0, L1 = m.layers[0], m.layers[1]
tri = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
same = (x[:, :, None] == x[:, None, :])
def shifted(M, d):  # M[r, i, j] -> mask true where x[j-d]==x[i]
    o = torch.zeros_like(M)
    if d > 0: o[:, :, d:] = M[:, :, :-d]
    elif d < 0: o[:, :, :d] = M[:, :, -d:]
    else: o = M.clone()
    return o
masks = {"x[j-1]==x[i] (induction)": shifted(same, 1), "x[j]==x[i] (same token)": same, "x[j-2]==x[i]": shifted(same, 2)}
P = m.pos(torch.arange(T)); E = m.emb(x)
def layer0(tok, pos, vpos=None, vtok=None):
    """Residual stream after layer 0 (+MLP0). vpos/vtok replace the inputs to L0H{H0}'s VALUES only (None = unchanged)."""
    h0 = tok + pos
    b, t, _ = h0.shape
    q, k, v = L0.qkv(h0).view(b, t, 3, H, D // H).permute(2, 0, 3, 1, 4)
    if vpos is not None or vtok is not None:
        hv = (tok if vtok is None else vtok) + (pos if vpos is None else vpos)
        v = v.clone(); v[:, H0] = L0.qkv(hv).view(b, t, 3, H, D // H).permute(2, 0, 3, 1, 4)[2][:, H0]
    a = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(tri, -1e9).softmax(-1)
    h = h0 + L0.o((a @ v).transpose(1, 2).reshape(b, t, D))
    return h + m.mlps[0](h) if MLPF else h
def report(name, hq, hk):
    q = L1.qkv(hq).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4)[0][:, H1]
    k = L1.qkv(hk).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4)[1][:, H1]
    A = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(tri, -1e9).softmax(-1)
    out = " | ".join(f"{k_.split(' ')[0]} {(A * (M & causal)).sum(-1)[cm].mean().item():.2f}" for k_, M in masks.items())
    print(f"{name:44s}: {out}")
def shiftpos(d):  # pos(j+d), clamped at the ends
    return P[(torch.arange(T) + d).clamp(0, T - 1)]
with torch.no_grad():
    clean = layer0(E, P)
    print(f"L0H{H0} -> L1H{H1}; columns: mass on x[j-1]==x[i] (induction target) | x[j]==x[i] | x[j-2]==x[i]")
    report("clean", clean, clean)
    print("-- keys only (queries clean)")
    g = torch.Generator().manual_seed(7); perm = torch.randperm(T, generator=g)
    report("K: positions permuted", clean, layer0(E, P[perm]))
    for d in (-1, 1, 2):
        report(f"K: positions shifted pos(j{d:+d})", clean, layer0(E, shiftpos(d)))
    xr = torch.randint(0, V, x.shape, generator=g)
    report("K: tokens replaced by random", clean, layer0(m.emb(xr), P))
    report("K: tokens zeroed", clean, layer0(torch.zeros_like(E), P))
    report("K: positions zeroed", clean, layer0(E, torch.zeros_like(P)))
    print("-- keys only, DIRECT term only: edit the embedding added to the residual, layer 0 + MLP0 left clean")
    report("K direct: positions permuted", clean, clean - P + P[perm])
    for d in (-1, 1):
        report(f"K direct: positions shifted pos(j{d:+d})", clean, clean - P + shiftpos(d))
    report("K direct: tokens replaced by random", clean, clean - E + m.emb(xr))
    report("K direct: positions removed", clean, clean - P)
    print(f"-- queries only (keys clean): inputs to L0H{H0}'s values")
    report(f"Q: L0H{H0} values from token emb only", layer0(E, P, vpos=torch.zeros_like(P)), clean)
    report(f"Q: L0H{H0} values from position emb only", layer0(E, P, vtok=torch.zeros_like(E)), clean)
    for d in (-1, 1):
        report(f"Q: L0H{H0} value positions shifted pos(j{d:+d})", layer0(E, P, vpos=shiftpos(d)), clean)
    report(f"Q: L0H{H0} value positions permuted", layer0(E, P, vpos=P[perm]), clean)
