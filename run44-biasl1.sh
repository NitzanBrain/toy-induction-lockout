# run44: causal test (PREREG-run44.md). A: classic seeds pushed toward j0+2; B: one-step-early seeds pushed toward j0+1.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for spec in "39 2" "45 2" "40 2" "47 1" "36 1" "44 1"; do
  set -- $spec
  echo "== RANDSTART L=4 seed $1, 8000 steps, BIASL1=0.5 OFF=$2 STEPS<2500"
  BIASL1=0.5 BIASL1_OFF=$2 BIASL1_STEPS=2500 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s$1_bl1off$2.pt $PY -I train.py 4 8000 $1 2>&1 | grep -E "^step|FINAL|saved"
  echo "-- qkv (L1 masses)"
  RANDSTART=1 BIGRAM=0.5 $PY -I analyze_qkv.py models/R4s$1_bl1off$2.pt 0 2>&1 | grep -E "^L1H|copy loss"
done
