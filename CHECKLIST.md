# Reproducibility checklist and verification record

## Verified before release (Nov 25, 2026)

| Check | Result |
|---|---|
| Full VAE-Mamba parameters | 7,892,545 (`tests/test_models.py`) |
| Compact VAE-Mamba parameters | 1,989,665 |
| Checkpoint compatibility | Parameter names and shapes identical to the original `vae_mamba.py` (stored in `tests/original_state_dict_shapes.json`); 24 checkpoints written by the original code load with `strict=True` |
| Training equivalence | On a toy dataset (CPU, 2 epochs, fixed seed), `train.py` and the original `vae_mamba.py` give bit-identical weights for the baseline and for a teacher run; `train_physkd.py` and the original `kd_student.py --fusion spatial` give bit-identical student, fusion and adapter weights for V1, V3 and the ResAug control, and the same `kd` record |
| Evaluation equivalence | `evaluate.py` and the original `vae_mamba.py --mode eval` write byte-identical metric CSVs and prediction files for the same checkpoint |
| Metric implementation | `physkd/metrics.py` is the original `ldct_metrics.py`; `physkd/nqm.py` is the NQM file used for every method of the paper (unchanged) |
| Configurations | `tests/test_protocol.py` checks lr 1e-4, batch 8, crop 256, 100 epochs, KL 1e-6 / 10 epochs, width 16 students, (lambda_pix, lambda_feat) per variant, ResAug = lambda_kd 0 + synthetic reconstruction |
| Data split | Train L067 L096 L109 L143 L192 L286 L291 L310, validation L333, test L506; disjoint; no config overrides it |
| Entry points | Every command of the README exists; `--help` works for all Python entry points; the full pipeline (teachers, baselines, V1/V2/V3, ResAug, resume, evaluation with export, tables) runs on a toy dataset in `tests/test_pipeline.py` |
| Synthesis | `make_synthetic.py` runs on a real Mayo-format slice and records its settings in `meta.json` |
| Missing results | `make_tables.py` stops on missing evaluation files; `evaluate.py` stops on missing checkpoints, targets or empty test sets |
| Private data | No DICOM, `.npy`, checkpoint, credentials or Mayo2020 patient identifier in the repository; `.gitignore` excludes data and runs |

## Not tested here

- A complete training run on Mayo2016 (only toy runs on CPU); GPU runs are seeded but not bitwise deterministic.
- Loading the actual paper checkpoints (only checkpoints of the same code on toy data were available).
- LPIPS / DISTS with their real pretrained weights (the sandbox had no network access; locally cached stand-in weights
  were used, which only exercises the code path). The identity self-check runs at every evaluation.
- `prepare_mayo.py` and `get_spacing.py` on real DICOM files (unchanged from the scripts used for the paper).

## To verify before making the repository public

1. Teacher training budget: the selected checkpoints are at epoch index 35, but the configured number of epochs of
   the original teacher runs is not recorded. `configs/teacher_*.yaml` use the script default (100).
2. Teacher training flags: the configs assume `--syn_ratio 1 --syn_families <family> --val_syn_dir <L333 set>` with the
   full model (width 32). Compare with the original commands or logs.
3. Residual synthesis: check `"residual"` in `meta.json` of the synthetic folders used for the teachers and for
   distillation, and state it in the paper and the configs (`SYN_ARGS="--residual"` in `scripts/prepare_data.sh`).
4. Order of the teachers in `--teachers` (fusion channel order) used in the paper runs; the configs use noise, streak,
   blur.
5. Size of the synthetic batch: configs use the default (`kd_syn_batch` = batch size = 8), as stated in the paper.
6. ResAug control: confirm it was trained with `kd_student.py --lambda_kd 0 --kd_syn_rec --kd_syn_extra` (same batches
   as PhysKD) rather than `vae_mamba.py --syn_dir`.
7. SD convention of the reported tables (ddof 0 or 1 across seeds); `make_tables.py` uses ddof = 1 across seeds and
   ddof = 0 between slices. Confirm whether the Mayo2020 numbers in `results/paper/` were aggregated per patient.
8. Mayo2020 anatomy, dose and slice thickness from the DICOM headers before describing them in the README.
9. Origin and license of `physkd/nqm.py`, and the license of the repository itself (none is included yet).
10. DenoMamba NQM value in `results/paper/` (0.89 dB) is on a different scale; it is excluded from the paper.
11. Add `docs/framework.png` and the checkpoint download link.
