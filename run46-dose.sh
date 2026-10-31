# run46 (PREREG-run46.md): unbiased baseline at 8000 + BIASL1 dose 0.1 / 0.25 toward j0+2 on classic seeds s39, s45, s40.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for b in 0 0.1 0.25; do for s in 39 45 40; do
  echo "== RANDSTART L=4 seed $s, 8000 steps, BIASL1=$b OFF=2 STEPS<2500"
  BIASL1=$b BIASL1_OFF=2 BIASL1_STEPS=2500 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}_bl1d$b.pt $PY -I train.py 4 8000 $s 2>&1 | grep -E "^step|FINAL|saved"
  echo "-- qkv (L1 masses)"
  RANDSTART=1 BIGRAM=0.5 $PY -I analyze_qkv.py models/R4s${s}_bl1d$b.pt 0 2>&1 | grep -E "^L1H|copy loss"
done; done
