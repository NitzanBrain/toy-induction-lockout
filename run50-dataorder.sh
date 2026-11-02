# run50 (PREREG-run50.md): same init, different training-batch order (DATA_SEED). No bias.
cd /home/brain/brain/projects/toy-induction
PY=/home/brain/brain/cache/venv/bin/python
for s in 46 35 32 33; do
  for d in 1001 1002 1003; do
    echo "== RANDSTART L=4 init seed $s, DATA_SEED $d, 8000 steps, no bias"
    DATA_SEED=$d HEADLOG=1 HEADLOG2=1 RANDSTART=1 MLP=1 BIGRAM=0.5 SAVE=/home/brain/brain/cache/toy/R4s${s}_data$d.pt $PY -I train.py 4 8000 $s 2>&1 | grep -E "^step|L1 early|FINAL|saved"
    echo "-- qkv (L1 masses)"
    RANDSTART=1 BIGRAM=0.5 $PY -I analyze_qkv.py /home/brain/brain/cache/toy/R4s${s}_data$d.pt 0 2>&1 | grep -E "^L1H|copy loss"
  done
done
echo RUN50 DONE
