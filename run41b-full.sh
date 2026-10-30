# run41b: full 12000-step training + route analysis for the NEW seeds 20-31 (outcome for PREREG-run41; predictions sealed in PREDICTIONS-run41.txt before this started).
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 20 21 22 23 24 25 26 27 28 29 30 31; do
  echo "== RANDSTART L=4 seed $s, 12000 steps"
  RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}L.pt $PY -I train.py 4 12000 $s 2>&1
  echo "-- routes"
  RANDSTART=1 BIGRAM=0.5 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}L.pt 2>&1 | head -40
done
