# run48 (PREREG-run47.md): A = BIASL1 0.05 → j0+2 on classic seeds; B = BIASL1 0.1 → j0+1 on one-step-early seeds.
cd /home/brain/brain/projects/toy-induction
PY=/home/brain/brain/cache/venv/bin/python
for spec in "47 1 0.25" "36 1 0.25" "44 1 0.25"; do
  set -- $spec
  echo "== RANDSTART L=4 seed $1, 8000 steps, BIASL1=$3 OFF=$2 STEPS<2500"
  BIASL1=$3 BIASL1_OFF=$2 BIASL1_STEPS=2500 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=/home/brain/brain/cache/toy/R4s$1_bl1off$2_d$3.pt $PY -I train.py 4 8000 $1 2>&1 | grep -E "^step|FINAL|saved"
  echo "-- qkv (L1 masses)"
  RANDSTART=1 BIGRAM=0.5 $PY -I analyze_qkv.py /home/brain/brain/cache/toy/R4s$1_bl1off$2_d$3.pt 0 2>&1 | grep -E "^L1H|copy loss"
done
