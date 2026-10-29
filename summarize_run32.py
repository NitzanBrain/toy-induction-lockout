# Per seed: escape step (first eval with copy loss < 1.0), final copy loss, and route per induction head
# (largest relative drop of target mass from removing one earlier head from Q only vs K only). Seed route = the single
# biggest relative drop over all its heads (the pre-registered "strongest single input"); per-head routes listed too.
# Run: python3 -I summarize_run32.py run32-seeds-escape.txt
import re, sys
seeds = {}; cur = None; head = None
for l in open(sys.argv[1]):
    m = re.match(r'== RANDSTART L=4 seed (\d+)', l)
    if m: cur = int(m.group(1)); seeds[cur] = dict(esc=None, final=None, heads={}); continue
    if cur is None: continue
    m = re.match(r'step +(\d+) .*copies ([\d.]+)', l)
    if m:
        st, c = int(m.group(1)), float(m.group(2)); seeds[cur]['final'] = c
        if c < 1.0 and seeds[cur]['esc'] is None: seeds[cur]['esc'] = st
    m = re.match(r'induction head (\S+): clean target mass ([\d.]+)', l)
    if m: head = m.group(1); seeds[cur]['heads'][head] = dict(clean=float(m.group(2)), Q=1.0, K=1.0, qs='', ks='')
    m = re.match(r'\s+remove (\S+): from Q -> ([\d.]+) \| from K -> ([\d.]+)', l)
    if m:
        h = seeds[cur]['heads'][head]; q, k = float(m.group(2)) / h['clean'], float(m.group(3)) / h['clean']
        if q < h['Q']: h['Q'], h['qs'] = q, m.group(1)
        if k < h['K']: h['K'], h['ks'] = k, m.group(1)
for s, d in sorted(seeds.items()):
    per = [f"{n}({h['clean']:.2f}): {'Q' if h['Q'] < h['K'] else 'K'} via {h['qs'] if h['Q'] < h['K'] else h['ks']} ({min(h['Q'], h['K']):.2f})" for n, h in d['heads'].items()]
    best = min(((min(h['Q'], h['K']), 'Q' if h['Q'] < h['K'] else 'K') for h in d['heads'].values()), default=(None, 'none'))
    print(f"s{s}: escape {d['esc']} final {d['final']} | seed route {best[1]} | " + "; ".join(per))
