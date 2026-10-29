# run37 (converse causal test): the 6 Q seeds with a +2 logit bias toward key i−1 in all L0 heads for steps < 2500, then
# none. Prediction if the early prev lean decides the route: they switch to classic K-composition (and escape earlier).
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 1 4 7 11 13 15; do
  echo "== RANDSTART L=4 seed $s, 10000 steps, BIASPREV_L0=2 until 2500"
  BIASPREV_L0=2 BIASPREV_STEPS=2500 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}B.pt $PY -I train.py 4 10000 $s 2>&1 | grep -vE "^  HEADS"
  echo "-- routes"
  RANDSTART=1 BIGRAM=0.5 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}B.pt 2>&1 | head -40
done
