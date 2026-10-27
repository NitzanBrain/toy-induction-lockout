# Reviewer caveat (10-27): every training copy starts at position 0, so absolute position is unusually informative.
# Does copying still work when the first segment starts at position o > 0 (filler before it)? Copy loss on the second
# copy and the induction head's target mass, for o in a few values. Segment length s in 8..(64-o)//2; control rows at o=0
# with the shorter s ranges of o=4 (s<=30) and o=20 (s<=22) separate the start-position effect from shorter copies.
# Run: <venv>/bin/python -I analyze_offset.py <model.pt> [layer] [head]
import sys
src = open(__file__.replace("analyze_offset.py", "analyze_copy.py")).read().split("base = copy_loss()")[0]
LI = int(sys.argv[2]) if len(sys.argv) > 2 else 1; HI = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.argv = sys.argv[:2]; exec(src)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool), -1)
def build(o, seed=99, smax=None):
    torch.manual_seed(seed); xx = torch.randint(0, V, (N, T)); SS = torch.randint(8, (smax or (T - o) // 2) + 1, (N,))
    if float(os.environ.get("BIGRAM", 0)):
        for t_ in range(2, T):
            use = torch.rand(N) < P; xx[use, t_] = NEXT[xx[use, t_ - 2]]
    cmm = torch.zeros(N, T, dtype=torch.bool)
    for r in range(N):
        s = SS[r].item(); xx[r, o + s:o + 2 * s] = xx[r, o:o + s]; cmm[r, o + s:o + 2 * s - 1] = True
    return xx, cmm
for o, smax in ((0, None), (0, 30), (0, 22), (4, None), (10, None), (20, None)):
    x, cm = build(o, smax=smax)
    with torch.no_grad():
        lg = m(x); ce = F.cross_entropy(lg[:, :-1].transpose(1, 2), x[:, 1:], reduction="none")
    same = (x[:, :, None] == x[:, None, :]); prevm = torch.zeros_like(same); prevm[:, :, 1:] = same[:, :, :-1]
    A = m.layers[LI].pat[:, HI]
    print(f"copy starts at {o:2d}, s<= {smax or (T - o) // 2}: copy loss {ce[cm[:, :-1]].mean().item():.3f} | L{LI}H{HI} target mass {(A * (prevm & causal)).sum(-1)[cm].mean().item():.2f}")
