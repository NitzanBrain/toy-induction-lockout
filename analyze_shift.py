# Prediction test: the "position/2 pointer" solution should break when the repeated segment does not start at position 0,
# while a classic induction model (prev-token + induction) should not care. Copy loss vs start offset o of the first copy.
import sys, torch, torch.nn.functional as F
for path in sys.argv[1:]:
    sys.argv = [sys.argv[0], path]; exec(open("analyze_copy.py").read().split("base = copy_loss()")[0])
    res = []
    for o in [0, 2, 4, 8, 12]:
        torch.manual_seed(5); n = 512; xx = torch.randint(0, V, (n, T)); SS = torch.randint(8, 25, (n,)); mk = torch.zeros(n, T, dtype=torch.bool)
        for r in range(n):
            s = SS[r].item(); xx[r, o + s:o + 2 * s] = xx[r, o:o + s]; mk[r, o + s:o + 2 * s - 1] = True
        with torch.no_grad(): lg = m(xx); ce = F.cross_entropy(lg[:, :-1].transpose(1, 2), xx[:, 1:], reduction="none")
        res.append((o, round(ce[mk[:, :-1]].mean().item(), 2)))
    print(path.split("/")[-1], "copy loss by start offset:", res)
