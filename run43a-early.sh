# run43a: 1500-step HEADLOG2 (4-decimal) for NEW seeds 32-47 → marker for PREREG-run43.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in $(seq 32 47); do
  echo "== RANDSTART L=4 seed $s, 1500 steps, HEADLOG2"
  HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5 $PY -I train.py 4 1500 $s 2>&1 | grep -E "^step|HEADS"
done
