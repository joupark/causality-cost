#!/bin/sh
# Reproduces the second (larger, MS-SNSD) trained-network test of
# Theorem 1 from the shipped best_big.pt checkpoint (no retraining).
set -e
export MSNSD_DIR="${MSNSD_DIR:-$(dirname "$0")/../msnsd_validate/MS-SNSD}"
python3 eval_context_big.py --batch 1
python3 eval_context_big.py --batch 2,4,8
python3 eval_context_big.py --batch 16,32,64
python3 eval_context_big.py --batch 128,None
python3 - <<'PY'
import pandas as pd, glob
files = [f for f in sorted(glob.glob('results_context_big_*.csv')) if 'mse' not in f]
pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv('results_context_big.csv', index=False)
mfiles = sorted(glob.glob('results_context_big_mse_*.csv'))
pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True).to_csv('results_context_big_mse.csv', index=False)
PY
python3 eval_context_big.py --batch analyze
