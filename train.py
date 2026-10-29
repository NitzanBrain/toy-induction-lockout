# Train tiny attention-only transformers from scratch on CPU and watch induction heads form (Olsson et al. 2022 /
# Elhage et al. 2021 toy setting). Data: sequences of random tokens where a random half-length segment is repeated,
# so in-context copying pays off. Compare 1-layer vs 2-layer (only ≥2 layers can form induction heads via K-composition).
# Logged every K steps: loss on the 2nd copy vs 1st, the max prev-token score and max induction score over heads.
# Run: <venv>/bin/python -I train.py <n_layers> <steps> [seed]
import sys, math, torch, torch.nn as nn, torch.nn.functional as F
NL, STEPS = int(sys.argv[1]), int(sys.argv[2]); SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(SEED); torch.set_num_threads(8)
import os; BIGRAM = float(os.environ.get('BIGRAM', 0)); MLP = os.environ.get('MLP') == '1'; NEXT = torch.randperm(64, generator=torch.Generator().manual_seed(123))  # fixed bigram map
V, T, D, B = 64, 64, 64, 64; H = int(__import__("os").environ.get("HEADS", 4))
class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv = nn.Linear(D, 3 * D, bias=False); s.o = nn.Linear(D, D, bias=False)
    def forward(s, x):
        b, t, _ = x.shape; q, k, v = s.qkv(x).view(b, t, 3, H, D // H).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / math.sqrt(D // H); a = a.masked_fill(torch.triu(torch.ones(t, t, dtype=torch.bool), 1), -1e9)
        if getattr(s, "maskprev", False): a = a.masked_fill(torch.diag(torch.ones(t - 1, dtype=torch.bool), -1), -1e9)  # forbid key i−1
        if getattr(s, "biasprev", 0): a = a + s.biasprev * torch.diag(torch.ones(t - 1), -1)  # converse test: push toward key i−1
        a = a.softmax(-1)
        s.pat = a.detach(); return s.o((a @ v).transpose(1, 2).reshape(b, t, D))
class Model(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(T, D); s.layers = nn.ModuleList(Attn() for _ in range(NL)); s.un = nn.Linear(D, V, bias=False)
        s.mlps = nn.ModuleList(nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) for _ in range(NL))
    def forward(s, x):
        h = s.emb(x) + s.pos(torch.arange(x.shape[1]))
        for k, l in enumerate(s.layers):
            h = h + l(h)
            if MLP: h = h + s.mlps[k](h)  # optional MLP after each attention layer (env MLP=1)
        return s.un(h)
def batch(n=B):
    # random tokens; a segment of random length s ∈ [8, 32] is followed immediately by a copy of itself, then random filler.
    # The copy's offset s varies per sample, so a purely positional head can't solve it (my first version had a fixed
    # offset T/2 and the model learned a positional shortcut: "induction" 0.87 with prev-token 0.04).
    x = torch.randint(0, V, (n, T)); S = torch.randint(8, 33, (n,))
    if BIGRAM:  # skip-one structure: with prob BIGRAM, x[t] = NEXT[x[t-2]]. Predicting x[t+1] then needs the token
        # BEFORE the current one, i.e. a prev-token head pays off on its own (a plain bigram x[t]=f(x[t-1]) wouldn't:
        # the direct embedding→unembedding path handles it without attention).
        for t_ in range(2, T):
            use = torch.rand(n) < BIGRAM; x[use, t_] = NEXT[x[use, t_ - 2]]
    # RANDSTART=1: the first segment starts at a random position o ∈ [0, T−2s] instead of 0 (filler before it), so absolute
    # position no longer tells where copies are (10-27: models trained with o=0 fail when o>0, run26).
    O = torch.stack([torch.randint(0, T - 2 * s + 1, ()) for s in S.tolist()]) if os.environ.get("RANDSTART") == "1" else torch.zeros(n, dtype=torch.long)
    for r in range(n):
        s, o = S[r].item(), O[r].item(); x[r, o + s:o + 2 * s] = x[r, o:o + s]
    batch.S = S; batch.O = O; return x
m = Model(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
_b = BIGRAM; BIGRAM = float(os.environ.get("P_EVAL", BIGRAM))  # eval set's shortcut strength (default: the initial p)
evx = batch(256); evS = batch.S.clone(); evO = batch.O.clone(); BIGRAM = _b; j = torch.arange(1, T)
cmask = torch.zeros(256, T, dtype=torch.bool); fmask = torch.zeros(256, T, dtype=torch.bool)
for r in range(256):
    s, o = evS[r].item(), evO[r].item(); cmask[r, o + s:o + 2 * s - 1] = True; fmask[r, o + 2 * s:T - 1] = True  # positions predicting copied / filler tokens
SWITCH = int(os.environ.get("SWITCH", 0)); P2 = float(os.environ.get("P2", 0))
ANNEAL = os.environ.get("ANNEAL") == "1"; P0 = BIGRAM; P_END = float(os.environ.get("P_END", 0))  # curriculum: change p at step SWITCH
for step in range(STEPS + 1):
    if ANNEAL:  # linear anneal of the shortcut strength from the starting p to P_END over the whole run
        BIGRAM = P0 + (P_END - P0) * step / STEPS
    if SWITCH and step == SWITCH:
        BIGRAM = P2; print(f"--- switched shortcut strength to {P2} at step {step}", flush=True)
        if os.environ.get("RESET_QK"):  # Nikishin/Lyle-style reset: re-init the LAST layer's Q and K, and clear the optimiser state
            with torch.no_grad():
                w = m.layers[-1].qkv.weight; fresh = nn.Linear(D, 3 * D, bias=False).weight; w[:2 * D] = fresh[:2 * D]
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); print("--- reset last-layer Q,K + optimiser", flush=True)
    m.layers[0].maskprev = step < int(os.environ.get("MASKPREV_L0", 0))  # causal test (run36): no prev-token attention in L0 early
    m.layers[0].biasprev = float(os.environ.get("BIASPREV_L0", 0)) if step < int(os.environ.get("BIASPREV_STEPS", 0)) else 0  # run37: the converse
    x = batch(); lg = m(x); loss =F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % (STEPS // 80) == 0:
        with torch.no_grad():
            lg = m(evx); ce = F.cross_entropy(lg[:, :-1].transpose(1, 2), evx[:, 1:], reduction="none")
            l2, lf = ce[cmask[:, :-1]].mean().item(), ce[fmask[:, :-1]].mean().item()
            prev = max(l.pat[:, h][:, j, j - 1].mean().item() for l in m.layers for h in range(H))
            ind = 0.0
            for l in m.layers:
                for h in range(H):
                    sc = [l.pat[r, h, torch.arange(o + s, o + 2 * s - 1), torch.arange(o + 1, o + s)].mean().item() for r, (s, o) in enumerate(zip(evS.tolist(), evO.tolist()))]
                    ind = max(ind, sum(sc) / len(sc))
        print(f"step {step:6d} train {loss.item():.3f} | loss on copies {l2:.3f} on filler {lf:.3f} | max prev-token {prev:.2f} max induction {ind:.2f}", flush=True)
        if os.environ.get("HEADLOG"):  # per-head L0 prev-token, L0 same-token (offset −s) and L1 induction scores, to see which forms first
            l0, l1 = m.layers[0], m.layers[1]
            pv = [round(l0.pat[:, h][:, j, j - 1].mean().item(), 2) for h in range(H)]
            sv = [round(sum(l0.pat[r, h, torch.arange(o + s, o + 2 * s - 1), torch.arange(o, o + s - 1)].mean().item() for r, (s, o) in enumerate(zip(evS.tolist(), evO.tolist()))) / len(evS), 2) for h in range(H)]
            iv = [round(sum(l1.pat[r, h, torch.arange(o + s, o + 2 * s - 1), torch.arange(o + 1, o + s)].mean().item() for r, (s, o) in enumerate(zip(evS.tolist(), evO.tolist()))) / len(evS), 2) for h in range(H)]
            print(f"  HEADS L0 prev {pv} | L0 same {sv} | L1 ind {iv}", flush=True)
# final per-layer report: which layer holds the prev-token head, and the induction scores per layer
with torch.no_grad():
    m(evx)
    for li, l in enumerate(m.layers):
        pv = [round(l.pat[:, h][:, j, j - 1].mean().item(), 2) for h in range(H)]
        iv = [round(sum(l.pat[r, h, torch.arange(o + s, o + 2 * s - 1), torch.arange(o + 1, o + s)].mean().item() for r, (s, o) in enumerate(zip(evS.tolist(), evO.tolist()))) / len(evS), 2) for h in range(H)]
        dv = [round(sum(l.pat[r, h, torch.arange(o + s, o + 2 * s - 1), torch.arange(o, o + s - 1)].mean().item() for r, (s, o) in enumerate(zip(evS.tolist(), evO.tolist()))) / len(evS), 2) for h in range(H)]
        print(f"FINAL layer {li}: prev-token per head {pv} | induction per head {iv} | same-token (offset −s) per head {dv}")
# optional: where do heads look on copy positions? histogram of (i − argmax attention) relative to the segment length s
if os.environ.get("OFFSETS"):
    import collections
    with torch.no_grad():
        m(evx)
        for li, l in enumerate(m.layers):
            for h in range(H):
                c = collections.Counter()
                for r, (s, o) in enumerate(zip(evS.tolist(), evO.tolist())):
                    for i_ in range(o + s + 1, o + 2 * s - 1):
                        k = int(l.pat[r, h, i_].argmax()); d = i_ - k
                        c["s-1 (induction)" if d == s - 1 else "s (same token)" if d == s else "1 (prev)" if d == 1 else "0 (self)" if d == 0 else ("abs pos %d" % k if k < 3 else "s-1±2" if abs(d - (s - 1)) <= 2 else "rel %+d" % (d - s))] += 1
                tot = sum(c.values()); print(f"OFFSETS L{li}H{h}: " + ", ".join(f"{k} {v / tot:.2f}" for k, v in c.most_common(4)))
if os.environ.get("SAVE"):  # save weights for offline mechanism analysis (analyze_copy.py)
    torch.save({"state": m.state_dict(), "NL": NL, "H": H, "MLP": MLP}, os.environ["SAVE"]); print("saved", os.environ["SAVE"])
