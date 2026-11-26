#!/bin/bash
# Full and compact VAE-Mamba without distillation, and the compact ResAug control (needs the teachers' checkpoints
# only because it shares the training script of PhysKD-LDCT; with lambda_kd = 0 they do not supervise the student).
set -e
cd "$(dirname "$0")/.."
python train.py --config configs/full_baseline.yaml --out runs/full_baseline_s0 --seed 0
for s in ${SEEDS:-0 1 2}; do
  python train.py --config configs/small_baseline.yaml --out runs/small_baseline_s$s --seed $s
  python train_physkd.py --config configs/small_resaug.yaml --out runs/small_resaug_s$s --seed $s
done
