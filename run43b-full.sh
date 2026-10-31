# run43b: 12000-step training + analyze_qkv (L1 mass on j0+1 vs j0+2) for NEW seeds 32-47 — outcome for PREREG-run43 (markers sealed in PREDICTIONS-run43.txt before this started).
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in $(seq 32 47); do
  echo "== RANDSTART L=4 seed $s, 12000 steps"
  RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}L.pt $PY -I train.py 4 12000 $s 2>&1 | grep -E "^step|FINAL|saved"
  echo "-- qkv (L1 masses)"
  RANDSTART=1 BIGRAM=0.5 $PY -I analyze_qkv.py models/R4s${s}L.pt 0 2>&1 | grep -E "^L1H|copy loss"
  echo "-- routes"
  RANDSTART=1 BIGRAM=0.5 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}L.pt 2>&1 | head -40
done
