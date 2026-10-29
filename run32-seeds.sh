# run32: (1) determinism check: s4 at 6000 steps must reproduce run30 (copy 2.201 at 6000).
# (2) random-start 4L seeds 10-17 at 12000 steps, full logs -> escape step (first eval with copy loss < 1.0) vs route.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
echo "== CHECK s4 6000"
RANDSTART=1 MLP=1 BIGRAM=0.5 $PY -I train.py 4 6000 4 2>&1 | grep -E "step +(1500|6000) "
for s in 10 11 12 13 14 15 16 17; do
  echo "== RANDSTART L=4 seed $s, 12000 steps"
  RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}L.pt $PY -I train.py 4 12000 $s 2>&1
  echo "-- routes"
  RANDSTART=1 BIGRAM=0.5 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}L.pt 2>&1 | head -40
done
