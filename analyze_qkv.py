# Q/K/V path patching of one layer-0 head into ALL layer-1 heads at once (random-start 4L s0, run28: no classic
# induction head, L1 heads attend one past the classic target, L0H3 attends ~2 back). For each of Q, K, V of layer 1,
# rebuild it from the layer-0 stream with head L0H{H0} zeroed (MLP0 recomputed), keep the other two clean, run the rest
# of the model, report copy loss. Also: L0H{H0}'s mean attention by offset d = i − j (all positions), and L1 heads' mass
# at key j0+1 (classic target) vs j0+2 on copy positions.
# Run: [RANDSTART=1] BIGRAM=0.5 <venv>/bin/python -I analyze_qkv.py <model.pt> [layer0_head]
import sys
src = open(__file__.replace("analyze_qkv.py", "analyze_copy.py")).read().split("base = copy_loss()")[0]
H0 = int(sys.argv[2]) if len(sys.argv) > 2 else 3
sys.argv = sys.argv[:2]; exec(src)
L0, L1 = m.layers[0], m.layers[1]
tri = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
with torch.no_grad():
    h0 = m.emb(x) + m.pos(torch.arange(T))
    def after0(mask):
        L0.mask = mask; h = h0 + L0(h0); L0.mask = None
        return h + m.mlps[0](h) if MLPF else h
    clean = after0(None); mk = torch.ones(H); mk[H0] = 0; abl = after0(mk)
    A0 = L0.pat[:, H0]  # pattern from the last L0 call (ablation only masks outputs, pattern unchanged)
    offs = [round(A0.diagonal(offset=-d, dim1=1, dim2=2).mean().item(), 2) for d in range(0, 5)]
    print(f"L0H{H0} mean attention at offset d=0..4: {offs}")
    def split(h): return L1.qkv(h).view(N, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
    qc, kc, vc = split(clean); qa, ka, va = split(abl)
    def run(q, k, v):
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(D // H)).masked_fill(tri, -1e9).softmax(-1)
        h = clean + L1.o((a @ v).transpose(1, 2).reshape(N, T, D))
        if MLPF: h = h + m.mlps[1](h)
        for kk in range(2, NL):
            h = h + m.layers[kk](h)
            if MLPF: h = h + m.mlps[kk](h)
        ce = F.cross_entropy(m.un(h)[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
        return ce[cm[:, :-1]].mean().item(), a
    base, a = run(qc, kc, vc)
    same = (x[:, :, None] == x[:, None, :]); causal = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
    def sh(M, d): o = torch.zeros_like(M); o[:, :, d:] = M[:, :, :-d]; return o
    for hh in range(H):
        print(f"L1H{hh}: mass on j0+1 (classic target) {(a[:, hh] * (sh(same, 1) & causal)).sum(-1)[cm].mean():.2f} | on j0+2 {(a[:, hh] * (sh(same, 2) & causal)).sum(-1)[cm].mean():.2f} | on j0+3 {(a[:, hh] * (sh(same, 3) & causal)).sum(-1)[cm].mean():.2f}")  # j0+3 added for run45 (control)
    print(f"copy loss, L1 rebuilt clean (sanity): {base:.3f}")
    for name, (q, k, v) in {"L0H%d removed from layer-1 Q" % H0: (qa, kc, vc), "from K": (qc, ka, vc), "from V": (qc, kc, va),
                            "from Q, K and V": (qa, ka, va)}.items():
        print(f"copy loss, {name}: {run(q, k, v)[0]:.3f}")
    # "one step early" test: zero layer 1's attention output only at the copy positions i, or only at i−1
    z_full = L1.o((a @ vc).transpose(1, 2).reshape(N, T, D))
    def run_masked(pos_mask, eval_mask):
        h = clean + z_full * (~pos_mask).unsqueeze(-1).float()
        if MLPF: h = h + m.mlps[1](h)
        for kk in range(2, NL):
            h = h + m.layers[kk](h)
            if MLPF: h = h + m.mlps[kk](h)
        ce = F.cross_entropy(m.un(h)[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
        return ce[eval_mask[:, :-1]].mean().item()
    # zero layer 1's attention output at EVEN positions only; evaluate on even copy positions (own output zeroed, i−1
    # intact) vs odd copy positions (own output intact, i−1 zeroed). Then the same with odd/even swapped.
    par = (torch.arange(T) % 2 == 0).expand(N, T)
    for name, zm in (("even", par), ("odd", ~par)):
        print(f"layer-1 attn output zeroed at {name} positions: copy loss where own position zeroed {run_masked(zm, cm & zm):.3f} | where only i-1 zeroed {run_masked(zm, cm & ~zm):.3f} | nothing zeroed: {run_masked(torch.zeros_like(zm), cm):.3f}")
    # Single-position version (review 10-27): zero layer 1's output only at positions ≡ r (mod 8). Evaluate the copy loss
    # where the zeroed position is i itself, i−1, i−2, or none of i..i−7 is zeroed... (the last two are within the same mod-8 class
    # pattern). Averaged over r = 0..7.
    import collections
    res = collections.defaultdict(list)
    idx = torch.arange(T)
    for r in range(8):
        zm = (idx % 8 == r).expand(N, T)
        for lag in (0, 1, 2, 3):
            ev = cm & ((idx - lag) % 8 == r).expand(N, T)   # positions i where i−lag is the zeroed one
            res[lag].append(run_masked(zm, ev))
        res["clean"].append(run_masked(torch.zeros_like(zm), cm))
    print("layer-1 attn output zeroed at one position in 8; copy loss at i when the zeroed position is "
          + " | ".join(f"i-{lag}: {sum(v) / len(v):.3f}" for lag, v in res.items() if lag != "clean")
          + f" | clean {sum(res['clean']) / 8:.3f}")
