#!/bin/sh
# Reproduces the MS-SNSD second-corpus validation (Section V-E, Fig. 4).
# Requires the MS-SNSD test folders (see README.md for how to fetch them
# without downloading the ~2.5GB full repository).
set -e
export MSNSD_DIR="${MSNSD_DIR:-$(dirname "$0")/MS-SNSD}"
python3 msnsd_validate.py --start 0  --end 20 --out results_msnsd_a.csv
python3 msnsd_validate.py --start 20 --end 40 --out results_msnsd_b.csv
python3 msnsd_mse.py      --start 0  --end 40 --out results_msnsd_mse.csv
python3 msnsd_numbers.py
