# run39: threshold search, bias 0.1 and 0.25 (run38: 0.5 already flips 3/3).
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for b in 0.1 0.25; do for s in 7 11 13; do
  echo "== RANDSTART L=4 seed $s, 10000 steps, BIASPREV_L0=$b until 2500"
  BIASPREV_L0=$b BIASPREV_STEPS=2500 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=models/R4s${s}B$b.pt $PY -I train.py 4 10000 $s 2>&1
  echo "-- routes"
  RANDSTART=1 BIGRAM=0.5 THRESH=0.3 $PY -I analyze_route.py models/R4s${s}B$b.pt 2>&1 | head -40
done; done
