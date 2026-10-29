# run33: fine-grained escape steps for random-start 4L seeds 0-9 (routes already known from run27-31), 8000 steps, eval every 100.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "== RANDSTART L=4 seed $s, 8000 steps"
  RANDSTART=1 MLP=1 BIGRAM=0.5 $PY -I train.py 4 8000 $s 2>&1 | grep -E "^step"
done
