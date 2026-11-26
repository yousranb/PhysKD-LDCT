#!/bin/bash
# PhysKD-LDCT V1 (output), V2 (latent), V3 (joint), compact student, three seeds.
set -e
cd "$(dirname "$0")/.."
for v in 1 2 3; do
  for s in ${SEEDS:-0 1 2}; do
    python train_physkd.py --config configs/physkd_v$v.yaml --out runs/physkd_v${v}_s$s --seed $s
  done
done
