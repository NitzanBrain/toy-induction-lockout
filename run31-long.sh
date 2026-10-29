cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 7 1 4 5; do
  echo "== RANDSTART L=4 seed $s, 18000 steps"
  RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}L.pt $PY -I train.py 4 18000 $s 2>&1 | awk 'NR%10==1 || /FINAL|saved/'
  echo "-- routes"
  RANDSTART=1 BIGRAM=0.5 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}L.pt 2>&1 | head -40
done
