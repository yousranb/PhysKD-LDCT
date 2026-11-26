#!/bin/bash
# DICOM -> .npy for Mayo2016 (3 mm and 1 mm), pixel spacing, and the synthetic teacher/distillation sets.
# Expected layout: $MAYO2016/<patient>/{quarter_3mm,full_3mm,quarter_1mm,full_1mm}/*.IMA
set -e
cd "$(dirname "$0")/.."
MAYO2016=${MAYO2016:-data/Mayo2016}
WORKERS=${WORKERS:-$(nproc)}
[ -d "$MAYO2016" ] || { echo "Mayo2016 DICOM folder not found: $MAYO2016 (set MAYO2016=...)"; exit 1; }
python prepare_mayo.py --data_path "$MAYO2016" --save_path data/npy_3mm --mm 3
python prepare_mayo.py --data_path "$MAYO2016" --save_path data/npy_1mm --mm 1
python get_spacing.py "$MAYO2016" 3 > data/spacing.json
# synthetic sets from the full-dose slices: training patients (teachers, distillation) and the validation patient
python make_synthetic.py --npy_dir data/npy_3mm --out_dir data/syn_3mm --spacing_json data/spacing.json \
    --workers "$WORKERS" ${SYN_ARGS}
python make_synthetic.py --npy_dir data/npy_3mm --out_dir data/syn_3mm_val --spacing_json data/spacing.json \
    --patients L333 --workers "$WORKERS" ${SYN_ARGS}
# Unseen test sets (simulated 10 % / 5 % dose of L506; Mayo2020): same conversion, one folder per set, e.g.
#   python prepare_mayo.py --data_path <unseen_dicom_3mm> --save_path data/npy_unseen_3mm --mm 3
