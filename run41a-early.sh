# run41a: early L0 head scores for the NEW seeds 20-31 (1000 steps each), for the pre-registered prediction.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 20 21 22 23 24 25 26 27 28 29 30 31; do
  echo "== RANDSTART L=4 seed $s, 1000 steps, HEADLOG"
  HEADLOG=1 RANDSTART=1 MLP=1 BIGRAM=0.5 $PY -I train.py 4 1000 $s 2>&1
done
