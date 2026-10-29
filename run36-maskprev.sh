# run36 (causal test): K seeds with the highest early L0 prev lean, trained with L0 attention to key i−1 forbidden for
# steps < 2500, then free. Same seeds/data as before. Does the route switch to Q (same-token pointer)? Escape later?
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 14 2 9 10 6 16; do
  echo "== RANDSTART L=4 seed $s, 10000 steps, MASKPREV_L0=2500"
  MASKPREV_L0=2500 HEADLOG=1 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}M.pt $PY -I train.py 4 10000 $s 2>&1
  echo "-- routes"
  RANDSTART=1 BIGRAM=0.5 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}M.pt 2>&1 | head -40
done
