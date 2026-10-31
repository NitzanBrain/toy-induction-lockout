# run45: control (PREREG-run45.md): classic seeds pushed toward key j0+3 instead of j0+2.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 39 45 40; do
  echo "== RANDSTART L=4 seed $s, 8000 steps, BIASL1=0.5 OFF=3 STEPS<2500"
  BIASL1=0.5 BIASL1_OFF=3 BIASL1_STEPS=2500 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}_bl1off3.pt $PY -I train.py 4 8000 $s 2>&1 | grep -E "^step|FINAL|saved"
  echo "-- qkv (L1 masses)"
  RANDSTART=1 BIGRAM=0.5 $PY -I analyze_qkv.py models/R4s${s}_bl1off3.pt 0 2>&1 | grep -E "^L1H|copy loss"
done
