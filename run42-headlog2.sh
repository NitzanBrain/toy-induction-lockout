# run42 (EXPLORATORY): 1000-step early logs with HEADLOG2 (L0 two-back, L1 one-step-early per head) for all 30 known seeds,
# to look for an early marker of the one-step-early route (eQ seeds: s0, s16 = early K; s15, s20, s24, s30 = early Q). Any marker found here must be tested prospectively on new seeds.
cd "$(dirname "$0")"; mkdir -p models
PY=${PY:-python}
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 20 21 22 23 24 25 26 27 28 29 30 31; do
  echo "== RANDSTART L=4 seed $s, 1000 steps, HEADLOG2"
  HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5 $PY -I train.py 4 1000 $s 2>&1 | grep -E "^step|HEADS"
done
