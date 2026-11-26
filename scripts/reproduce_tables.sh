#!/bin/bash
# Mean +- SD tables from the per-slice CSVs (results/eval) -> results/tables/Table_<set>.csv
set -e
cd "$(dirname "$0")/.."
python make_tables.py --eval_dir results/eval --out results/tables "$@"
