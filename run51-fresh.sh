# run51 (PREREG-run51.md): 16 fresh seeds, default data order, no bias.
cd /home/brain/brain/projects/toy-induction
PY=/home/brain/brain/cache/venv/bin/python
for s in $(seq 48 63); do
  echo "== RANDSTART L=4 init seed $s, 8000 steps, no bias"
  HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=/home/brain/brain/cache/toy/R4s${s}_run51.pt $PY -I train.py 4 8000 $s 2>&1 | grep -E "^step|L1 early|FINAL|saved"
  echo "-- qkv (L1 masses)"
  RANDSTART=1 BIGRAM=0.5 $PY -I analyze_qkv.py /home/brain/brain/cache/toy/R4s${s}_run51.pt 0 2>&1 | grep -E "^L1H|copy loss"
done
echo RUN51 DONE
