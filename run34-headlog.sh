# run34: when does each L0 head form relative to escape? Q seeds s7, s13 vs K seeds s5, s12 (random-start 4L, p=0.5, MLP), 7000 steps.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 7 13 5 12; do
  echo "== RANDSTART L=4 seed $s, 7000 steps, HEADLOG"
  HEADLOG=1 RANDSTART=1 MLP=1 BIGRAM=0.5 $PY -I train.py 4 7000 $s 2>&1
done
