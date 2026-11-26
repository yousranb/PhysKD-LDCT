# Changes with respect to the experimental code

The model, losses, sampling and evaluation are those of the original scripts; the training and evaluation entry
points were checked to give bit-identical results on a toy dataset (see CHECKLIST.md).

| Original | Here | Change |
|---|---|---|
| `vae_mamba.py` (model part) | `physkd/models.py` | Unchanged code; the unused SS2D (VMamba) variant removed; `build` rejects any block other than `mamba1` |
| `vae_mamba.py --mode train` | `train.py` | Unchanged training loop; removed options not used in the paper (`--init_ckpt`, `--save_every_epoch`, `--train_fraction`, SS2D); `--config` added |
| `vae_mamba.py --mode eval` | `evaluate.py` | Unchanged; clearer errors for a missing checkpoint |
| `kd_student.py` | `train_physkd.py`, `physkd/distill.py` | Unchanged training loop with the settings of the paper fixed: spatial fusion, constant KD weight, no selective KD, all real slices, model selection on real L333; removed image/equal fusion, KD decay, `--kd_select`, `--train_fraction`, synthetic validation and the old `--t_noise` syntax; student width default 16; `--config` added |
| `data_npy.py` + loaders of `vae_mamba.py` | `physkd/data.py` | Unchanged functions gathered in one module; split constants added |
| `ct_degradations.py` | `physkd/degradations.py` | Unchanged (header comment added) |
| `make_synthetic.py`, `get_spacing.py`, `prep_npy.py` | same names, `prepare_mayo.py` | Unchanged except imports, paths in the usage examples, and an unused argument removed |
| `ldct_metrics.py`, `nqm.py` | `physkd/metrics.py`, `physkd/nqm.py` | Unchanged except the import path of NQM |
| tables assembled by hand | `make_tables.py` | New: mean ± SD over seeds from per-slice CSVs, per-patient averaging for Mayo2020, refuses missing files |
| - | `configs/`, `scripts/`, `tests/` | New |

Not included: the benchmark of other methods (RED-CNN, WGAN-VGG, CTformer, DenoMamba), the full-size exploratory runs
(`Full E1/E6`), DenoMamba reproduction files, and the scripts of experiments not reported in the paper.
