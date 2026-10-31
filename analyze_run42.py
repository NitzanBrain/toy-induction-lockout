# run42 (exploratory): early markers (steps 500–900, max over heads) per seed vs route class.
# Classes: K = classic K, Q = classic Q, eK = one step early via K, eQ = one step early via Q (from §5 and §5.7).
# Run: python3 -I analyze_run42.py run42-headlog2.txt [lo hi]   (run42b: 4-decimal log with an extra 'L1 classic' field)
import re, ast, sys
LO, HI = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (500, 900)
CLS = {**{s: 'K' for s in (2, 3, 5, 6, 8, 9, 10, 12, 14, 17, 21, 23, 25, 27, 28, 29, 31)},
       **{s: 'Q' for s in (1, 4, 7, 11, 13, 22, 26)}, 0: 'eK', 16: 'eK', 15: 'eQ', 20: 'eQ', 24: 'eQ', 30: 'eQ'}
D = {}; cur = None; st = None
for l in open(sys.argv[1]):
    m = re.match(r'== RANDSTART L=4 seed (\d+)', l)
    if m: cur = int(m.group(1)); D[cur] = {'prev': [], 'same': [], 'ind': [], 'tb': [], 'early': [], 'cl': []}; continue
    m = re.match(r'step +(\d+) ', l)
    if m: st = int(m.group(1)); continue
    if st is None or not LO <= st <= HI: continue
    m = re.match(r'  HEADS L0 prev (\[.*?\]) \| L0 same (\[.*?\]) \| L1 ind (\[.*?\])', l)
    if m:
        for k, g in zip(('prev', 'same', 'ind'), m.groups()): D[cur][k].append(max(ast.literal_eval(g)))
    m = re.match(r'  HEADS2 L0 twoback (\[.*?\]) \| L1 early (\[.*?\])(?: \| L1 classic (\[.*?\]))?', l)
    if m:
        for k, g in zip(('tb', 'early', 'cl'), m.groups()):
            if g: D[cur][k].append(max(ast.literal_eval(g)))
avg = {s: {k: sum(v) / len(v) for k, v in d.items() if v} for s, d in D.items()}
keys = tuple(k for k in ('prev', 'same', 'ind', 'tb', 'early', 'cl') if all(k in avg[s] for s in avg))
print('seed cls ' + ' '.join(f'{k:>6}' for k in keys))
for s in sorted(avg, key=lambda s: (CLS[s], s)):
    print(f's{s:<3} {CLS[s]:3} ' + ' '.join(f'{avg[s][k]:6.3f}' for k in keys))
def auc(a, b):  # P(score in b > score in a)
    return sum((y > x) + 0.5 * (y == x) for x in a for y in b) / (len(a) * len(b))
e = [s for s in avg if CLS[s] in ('eK', 'eQ')]; c = [s for s in avg if CLS[s] in ('K', 'Q')]
for k in keys:
    print(f'{k:6}: AUC(one-step-early higher than classic) = {auc([avg[s][k] for s in c], [avg[s][k] for s in e]):.2f}  (n_early={len(e)}, n_classic={len(c)})')
