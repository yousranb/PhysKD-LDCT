#!/bin/bash
# Evaluate every run on every test set; per-slice CSVs in results/eval/<run>/<set>.csv, predictions in results/pred.
# Test data (prepare_mayo.py outputs):
#   NPY_3MM, NPY_1MM        : Mayo2016 25 % (L506)
#   UNSEEN_3MM, UNSEEN_1MM  : simulated 10 % / 5 % of L506 (file prefixes L506_10, L506_5)
#   MAYO2020_NPY            : Mayo2020 pairs; MAYO2020_PATIENTS: their file prefixes (not stored in this repository)
set -e
cd "$(dirname "$0")/.."
NPY_3MM=${NPY_3MM:-data/npy_3mm}; NPY_1MM=${NPY_1MM:-data/npy_1mm}
UNSEEN_3MM=${UNSEEN_3MM:-data/npy_unseen_3mm}; UNSEEN_1MM=${UNSEEN_1MM:-data/npy_unseen_1mm}
RUNS=${RUNS:-"full_baseline_s0 $(for r in small_baseline small_resaug physkd_v1 physkd_v2 physkd_v3; do for s in 0 1 2; do printf "%s_s%s " $r $s; done; done)"}
ev() {  # run set npy_dir patients...
  local run=$1 set=$2 dir=$3; shift 3
  [ -f "results/eval/$run/$set.csv" ] && { echo "done: $run $set"; return; }
  python evaluate.py --ckpt runs/$run/model_best.pkl --npy_dir "$dir" --test_patients "$@" \
      --out results/eval/$run/$set.csv --save_dir results/pred/$run/$set --method_name "$run"
}
for run in $RUNS; do
  [ -f runs/$run/model_best.pkl ] || { echo "missing checkpoint: runs/$run/model_best.pkl"; exit 1; }
  ev $run L506_3mm_25 "$NPY_3MM" L506
  ev $run L506_1mm_25 "$NPY_1MM" L506
  ev $run L506_3mm_10 "$UNSEEN_3MM" L506_10
  ev $run L506_1mm_10 "$UNSEEN_1MM" L506_10
  ev $run L506_3mm_5 "$UNSEEN_3MM" L506_5
  ev $run L506_1mm_5 "$UNSEEN_1MM" L506_5
  if [ -n "$MAYO2020_NPY" ]; then
    ev $run Mayo2020_25 "$MAYO2020_NPY" ${MAYO2020_PATIENTS:?set MAYO2020_PATIENTS to the Mayo2020 file prefixes}
  fi
done
