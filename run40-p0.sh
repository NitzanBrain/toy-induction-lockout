# run40 (control): the 6 Q seeds (under p=0.5) trained WITHOUT the skip-one shortcut (BIGRAM=0). If the shortcut drains the
# early L0 prev lean, then without it these seeds should take classic K and escape early.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 1 4 7 11 13 15; do
  echo "== RANDSTART L=4 seed $s, 6000 steps, BIGRAM=0"
  RANDSTART=1 MLP=1 SAVE=models/R4s${s}P0.pt $PY -I train.py 4 6000 $s 2>&1
  echo "-- routes"
  RANDSTART=1 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}P0.pt 2>&1 | head -40
done
