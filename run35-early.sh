# run35: early L0 head scores (HEADLOG) for all 18 random-start 4L seeds, 3000 steps (before any escape except s9 ~1000, s2/s3 ~1500).
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 0 1 2 3 4 6 8 9 10 11 14 15 16 17; do
  echo "== RANDSTART L=4 seed $s, 3000 steps, HEADLOG"
  HEADLOG=1 RANDSTART=1 MLP=1 BIGRAM=0.5 $PY -I train.py 4 3000 $s 2>&1
done
