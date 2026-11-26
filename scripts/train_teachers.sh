#!/bin/bash
# The three degradation-specific teachers (full VAE-Mamba, 7.89 M each).
set -e
cd "$(dirname "$0")/.."
for fam in noise streak blur; do
  python train.py --config configs/teacher_$fam.yaml --out runs/teacher_$fam --seed ${SEED:-0}
done
