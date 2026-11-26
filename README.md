# [PhysKD-LDCT framework](docs/framework.png)

## Installation

Python 3.10 or later. The selective scan is written in plain PyTorch, so no custom CUDA kernels are needed.

```bash
git clone <repository-url> PhysKD-LDCT && cd PhysKD-LDCT
pip install -r requirements.txt
python -m pytest -q tests        # about one minute on CPU
```

LPIPS and DISTS download their pretrained weights to `~/.cache/torch` on first use.

## Data

Mayo2016 (AAPM Low Dose CT Grand Challenge) must DOWNLOADED from " https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/144799205162 " . Place the DICOM files as

```
data/Mayo2016/<patient>/quarter_3mm/*.IMA   full_3mm/*.IMA   quarter_1mm/*.IMA   full_1mm/*.IMA
```

and run

```bash
bash scripts/prepare_data.sh
```

This converts every slice to `.npy` on the fixed scale x = (HU + 1024) / 4096 (no windowing, no per-image
normalization), reads the pixel spacing, and generates the synthetic training sets with `make_synthetic.py`
(one random severity level per family and slice, training patients only, plus a separate set for the validation
patient). The settings of each synthetic set, including whether the reconstruction residual was added to the
full-dose image (`--residual`), are written to `meta.json` in the output folder.

Split: training L067, L096, L109, L143, L192, L286, L291, L310; validation and model selection L333; test L506.
All models are trained on 3 mm, 25 % dose data only and tested without retraining on L506 at 25 % (1 mm and
3 mm), on simulated 10 % and 5 % dose versions of L506 (1 mm and 3 mm, generated separately from each full-dose
thickness), and on Mayo2020. The simulated doses and Mayo2020 are converted with the same script
(`prepare_mayo.py`); they are not produced by the teacher degradation operators.

## Training

Every entry point accepts `--config` (a file in `configs/`); options given on the command line override it.
Runs resume from `model_last.pkl` when restarted with the same `--out`; `model_best.pkl` is the epoch with the
highest validation PSNR on L333. Seeds fix the initialization and data order; GPU kernels are not forced to be
deterministic, so repeated runs can differ slightly.

Teachers (denoising, de-streaking, sharpening):

```bash
bash scripts/train_teachers.sh
# python train.py --config configs/teacher_noise.yaml --out runs/teacher_noise
```

Full and compact VAE-Mamba without distillation, and the compact ResAug control:

```bash
bash scripts/train_baselines.sh
# python train.py --config configs/small_baseline.yaml --out runs/small_baseline_s0 --seed 0
# python train_physkd.py --config configs/small_resaug.yaml --out runs/small_resaug_s0 --seed 0
```

ResAug uses the same real and synthetic batches as PhysKD-LDCT, but the synthetic inputs are supervised by their
reference image and there is no distillation term (`lambda_kd: 0`).

PhysKD-LDCT V1, V2 and V3 (seeds 0, 1, 2):

```bash
bash scripts/train_physkd.sh
# python train_physkd.py --config configs/physkd_v1.yaml --out runs/physkd_v1_s0 --seed 0
```

| Variant | lambda_pix | lambda_feat |
|---|---|---|
| V1, output | 1.0 | 0.0 |
| V2, latent | 0.0 | 1.0 |
| V3, joint | 0.5 | 0.2 |

Common settings: Adam, learning rate 1e-4, 100 epochs, 8 real 256 x 256 crops per step plus 8 synthetic
degraded inputs, KL weight warmed up linearly to 1e-6 over 10 epochs, lambda_rec = lambda_kd = 1, spatial fusion
trained against the full-dose image, distillation terms rescaled to the magnitude of the reconstruction loss.
The three variants differ in their distillation objective; they are not a factorial ablation of one component.

## Evaluation

```bash
python evaluate.py --ckpt runs/physkd_v1_s0/model_best.pkl --npy_dir data/npy_3mm --test_patients L506 \
    --out results/eval/physkd_v1_s0/L506_3mm_25.csv --save_dir results/pred/physkd_v1_s0/L506_3mm_25
```

The script writes PSNR, SSIM, RMSE, VIF, LPIPS, DISTS and NQM for every test slice, for the LDCT input and for
the model, on full 512 x 512 slices and on the fixed [0,1] scale, and exports the predictions as `.npy`.
`scripts/evaluate.sh` runs all models on all test sets (paths are set with environment variables listed at the
top of the script; Mayo2020 file prefixes are passed with `MAYO2020_PATIENTS`).

## Reproducing the tables

```bash
bash scripts/reproduce_tables.sh
```

`make_tables.py` averages each run over the test slices (Mayo2020: per patient first, then over patients) and
reports the mean and standard deviation over the three seeds. For single runs the standard deviation is computed
between test slices, and the `SD_type` column says which one is reported. Missing evaluation files are listed and
stop the script.

## Results

The values below are those of the paper (`results/paper/`). PSNR in dB, L506; mean ± SD over three seeds unless
marked as a single run (no SD shown). Missing entries were not evaluated.

Model capacity, synthetic inputs and distillation strategy (3 mm):

| Configuration | 25 % | 10 % | 5 % |
|---|---|---|---|
| Full VAE-Mamba (7.89 M), single run | 44.72 | — | — |
| Compact VAE-Mamba (1.99 M) | 44.54 ± 0.18 | 41.29 ± 0.14 | 35.84 ± 0.53 |
| Compact + ResAug | 44.01 ± 0.16 | 41.22 ± 0.29 | 36.79 ± 0.33 |
| Compact + output KD (V1) | 43.97 ± 0.04 | 42.02 ± 0.17 | 37.85 ± 0.67 |
| Compact + latent KD (V2) | 44.48 ± 0.10 | 41.44 ± 0.16 | 36.40 ± 0.53 |
| Compact + joint KD (V3) | 43.89 ± 0.12 | 41.88 ± 0.46 | 37.20 ± 1.02 |

Comparison with other methods, trained and tested with the same split and metrics:

| Method | 3 mm, 25 % | 3 mm, 10 % | 3 mm, 5 % | 1 mm, 25 % | 1 mm, 10 % | 1 mm, 5 % |
|---|---|---|---|---|---|---|
| RED-CNN | 44.04 ± 0.05 | 39.83 ± 0.18 | 34.12 ± 0.24 | 39.59 ± 0.06 | 38.09 ± 0.12 | 33.25 ± 0.20 |
| WGAN-VGG | 42.61 ± 0.59 | 38.75 ± 0.25 | 34.07 ± 0.17 | 38.33 ± 0.52 | 37.45 ± 0.33 | 33.35 ± 0.20 |
| CTformer | 42.49 ± 0.01 | 35.74 ± 0.05 | 30.03 ± 0.03 | 37.77 ± 0.02 | 35.01 ± 0.04 | 29.65 ± 0.03 |
| DenoMamba (single run) | 44.73 | — | — | — | — | — |
| PhysKD-LDCT V1 | 43.97 ± 0.04 | 42.02 ± 0.17 | 37.85 ± 0.67 | 39.22 ± 0.08 | 39.06 ± 0.09 | 35.92 ± 0.51 |
| PhysKD-LDCT V2 | 44.48 ± 0.10 | 41.44 ± 0.16 | 36.40 ± 0.53 | 40.12 ± 0.08 | 38.96 ± 0.09 | 35.02 ± 0.43 |
| PhysKD-LDCT V3 | 43.89 ± 0.12 | 41.88 ± 0.46 | 37.20 ± 1.02 | 39.20 ± 0.16 | 39.03 ± 0.30 | 35.45 ± 0.78 |

Mayo2020 (models trained on Mayo2016, no retraining):

| Method | PSNR (dB) | SSIM | LPIPS (×10⁻²) | DISTS (×10⁻²) |
|---|---|---|---|---|
| RED-CNN | 32.75 ± 0.35 | 0.7646 ± 0.0124 | 13.43 ± 0.57 | 12.27 ± 0.50 |
| WGAN-VGG | 31.92 ± 0.21 | 0.7390 ± 0.0160 | 11.40 ± 0.39 | 10.88 ± 0.24 |
| CTformer | 28.24 ± 0.02 | 0.6114 ± 0.0011 | 23.63 ± 0.04 | 16.43 ± 0.02 |
| Compact VAE-Mamba | 34.71 ± 0.19 | 0.8068 ± 0.0074 | 12.01 ± 4.38 | 11.97 ± 2.33 |
| Compact + ResAug | 34.46 ± 0.15 | 0.7867 ± 0.0103 | 17.42 ± 1.06 | 16.33 ± 0.65 |
| PhysKD-LDCT V1 | 34.66 ± 0.06 | 0.7898 ± 0.0008 | 17.97 ± 1.68 | 16.50 ± 0.49 |
| PhysKD-LDCT V2 | 34.71 ± 0.22 | 0.8082 ± 0.0047 | 8.71 ± 1.16 | 10.12 ± 0.78 |
| PhysKD-LDCT V3 | 34.44 ± 0.39 | 0.7866 ± 0.0078 | 16.22 ± 1.57 | 15.75 ± 0.58 |

The tables in `results/paper/` also contain SSIM, VIF, LPIPS, DISTS and NQM, and the LDCT input values.

## Checkpoints

Checkpoints are PyTorch files containing `net_model` (state dict), `width` and `block`; distilled students also
store the fusion module, the latent adapter and the distillation settings (`kd`). They load with

```python
import torch
from physkd.models import build
ck = torch.load("model_best.pkl", map_location="cpu")
net = build(ck.get("block", "mamba1"), ck.get("width", 32)).eval()
net.load_state_dict(ck["net_model"])
```

Trained checkpoints of the paper: link to be added.

## Citation

```bibtex
@article{taifour2026physkd,
  author  = {Taifour, Yousra and Tliba, Marouane and Beghdadi, Azeddine and Zaidi, Habib and Durak, Gorkem and
             Ming, Zuheng and Luong, Marie and Alaya Cheikh, Faouzi and Penhaker, Marek and Bagci, Ulas and
             Chetouani, Aladine},
  title   = {Physics-Informed Degradation-aware Knowledge Distillation for Robust and Compact {LDCT} Enhancement},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  note    = {submitted},
  year    = {2026}
}

@inproceedings{taifour2026multiexpert,
  author    = {Taifour, Yousra and Tliba, Marouane and Beghdadi, Azeddine and Chetouani, Aladine and Ming, Zuheng and
               Luong, Marie and Zaidi, Habib},
  title     = {Multi-Expert Distillation for Degradation-Aware Low-Dose {CT} Enhancement},
  booktitle = {Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)},
  year      = {2026},
  pages     = {6266--6270},
  doi       = {10.1109/ICASSP55912.2026.11464452}
}
```

## Acknowledgments

We thank the Mayo Clinic, the American Association of Physicists in Medicine and the National Institute of
Biomedical Imaging and Bioengineering for the Low Dose CT Grand Challenge data, and the providers of the
Mayo2020 low-dose CT data set. Part of the experiments were carried out on the Grid'5000 testbed, supported by a
scientific interest group hosted by Inria and including CNRS, RENATER and several universities as well as other
organizations (see https://www.grid5000.fr).
