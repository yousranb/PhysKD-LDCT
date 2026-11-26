"""Synthetic degraded/clean training pairs from full-dose .npy slices (physkd/degradations.py).

For every full-dose slice  <P>_<i>_target.npy  of the TRAINING patients:
  clean  = clean_reference(FDCT)                      -> <P>_<i>_cleanref.npy       (target of all synthetic pairs)
  noise  = sim_noise (gamma of the chosen level)      -> <P>_<i>_noise<Lk>_input.npy
  blur   = sim_blur  (sigma of the chosen level)      -> <P>_<i>_blur<Lk>_input.npy
  streak = sim_streak(delta_L of the chosen level)    -> <P>_<i>_streak<Lk>_input.npy
All files on the fixed [0,1] scale, (HU + 1024) / 4096, float32, 512 x 512.

  --levels random : one random level (L0..L3) per family per slice  (3 inputs/slice, ~4 MB/slice)
  --levels all    : all 4 levels for every family                   (12 inputs/slice, ~13 MB/slice)
  --residual      : keep the REAL scanner texture: input = FDCT + (FBP(degraded) - FBP(clean)), target = the
                    original FDCT (saved as <P>_<i>_cleanref.npy so the loaders are unchanged). Only the simulated
                    degradation residual is added, so there is no FBP-texture gap between synthetic and real data.
  --residual_base ldct : (test sets) add the residual to the REAL LDCT input <P>_<i>_input.npy instead of the FDCT,
                    target = FDCT: real noise + simulated streak / blur, i.e. combined degradations.

Streak amplitude A0 is calibrated ONCE (calibrate_A0 on a middle slice of the first patient) and reused for all
slices; it is stored in <out_dir>/meta.json together with the spacing and all settings, including "residual",
which records how a given synthetic set was generated.
Run on a CPU node (embarrassingly parallel):
  python make_synthetic.py --npy_dir data/npy_3mm --out_dir data/syn_3mm --spacing_json data/spacing.json --workers 32
"""
import argparse
import glob
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np

from physkd import degradations as D

FAMILIES = {"noise": D.NOISE_LEVELS, "blur": D.BLUR_LEVELS, "streak": D.STREAK_LEVELS}
TRAIN_DEFAULT = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"]
G = {}


def to_hu(x):
    return x * 4096.0 - 1024.0


def to01(hu):
    return ((hu + 1024.0) / 4096.0).astype(np.float32)


def work(job):
    tgt, patient, idx = job
    a = G["args"]
    ds = G["spacing"].get(patient, a.default_spacing_cm)
    base = os.path.join(a.out_dir, f"{patient}_{idx}")
    if os.path.exists(base + "_done"):
        return 0
    rng = np.random.default_rng([a.seed, int(patient[1:]) if patient[1:].isdigit() else 0, idx])
    hu = to_hu(np.load(tgt).astype(np.float32))
    c = D.clean_reference(hu, ds)
    base_hu = to_hu(np.load(tgt.replace("_target.npy", "_input.npy")).astype(np.float32)) if a.residual_base == "ldct" else hu
    np.save(base + "_cleanref.npy", to01(hu) if a.residual else to01(c["hu"]))
    streak_model = None
    for fam, levels in FAMILIES.items():
        names = list(levels) if a.levels == "all" else [list(levels)[rng.integers(len(levels))]]
        for lv in names:
            if fam == "noise":
                out = D.sim_noise(c, ds, levels[lv]["gamma"], seed=int(rng.integers(2**31)))
            elif fam == "blur":
                out = D.sim_blur(c, ds, levels[lv]["sigma"])
            else:
                if streak_model is None:
                    streak_model = D.build_projection_error_field(c, ds)
                out = D.sim_streak(c, ds, levels[lv]["delta_L"], streak_model, G["A0"])
            deg = base_hu + (out["hu"] - c["hu"]) if a.residual else out["hu"]
            np.save(f"{base}_{fam}{lv}_input.npy", to01(deg))
    open(base + "_done", "w").close()
    return 1


def init(args, spacing, A0):
    G.update(args=args, spacing=spacing, A0=A0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_dir", required=True, help="folder with <P>_<i>_target.npy (full dose, fixed [0,1])")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--patients", nargs="+", default=TRAIN_DEFAULT, help="TRAINING patients only")
    ap.add_argument("--levels", choices=["random", "all"], default="random")
    ap.add_argument("--spacing_json", default=None, help='{"L067": 0.0664, ...} pixel spacing in cm')
    ap.add_argument("--default_spacing_cm", type=float, default=0.07)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N slices per patient")
    ap.add_argument("--residual", action="store_true", help="real FDCT + simulated degradation residual (see header)")
    ap.add_argument("--residual_base", choices=["fdct", "ldct"], default="fdct",
                    help="image the residual is added to (ldct = combined real noise + simulated degradation)")
    a = ap.parse_args()
    if a.residual_base == "ldct" and not a.residual:
        raise SystemExit("--residual_base ldct requires --residual")
    os.makedirs(a.out_dir, exist_ok=True)

    spacing = json.load(open(a.spacing_json)) if a.spacing_json else {}
    missing = [p for p in a.patients if p not in spacing]
    if missing:
        print(f"WARNING: no pixel spacing for {missing}, using {a.default_spacing_cm} cm")

    jobs = []
    for p in a.patients:
        pat = re.compile(rf"^{re.escape(p)}_(\d+)_target\.npy$")
        files = sorted((int(pat.match(os.path.basename(f)).group(1)), f)
                       for f in glob.glob(os.path.join(a.npy_dir, f"{p}_*_target.npy")) if pat.match(os.path.basename(f)))
        if not files:
            raise FileNotFoundError(f"no {p}_<i>_target.npy in {a.npy_dir}")
        if a.limit:
            files = files[:: max(1, len(files) // a.limit)][: a.limit]
        jobs += [(f, p, i) for i, f in files]

    meta_path = os.path.join(a.out_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        A0 = meta["A0"]
        if "residual" in meta and meta["residual"] != a.residual:
            raise SystemExit(f"{a.out_dir} was made with residual={meta['residual']}: use another --out_dir")
        print(f"reusing A0 = {A0:.4f} from {meta_path}")
    else:
        f0 = [j for j in jobs if j[1] == a.patients[0]]
        f0 = f0[len(f0) // 2]
        ds0 = spacing.get(f0[1], a.default_spacing_cm)
        c0 = D.clean_reference(to_hu(np.load(f0[0]).astype(np.float32)), ds0)
        A0 = D.calibrate_A0(c0, ds0, D.build_projection_error_field(c0, ds0))
        json.dump({"A0": A0, "calibration_slice": os.path.basename(f0[0]), "levels": a.levels,
                   "residual": a.residual, "residual_base": a.residual_base,
                   "target": "original FDCT" if a.residual else "FBP clean reference",
                   "patients": a.patients, "spacing_cm": spacing, "default_spacing_cm": a.default_spacing_cm,
                   "noise": D.NOISE_LEVELS, "blur": D.BLUR_LEVELS, "streak": D.STREAK_LEVELS,
                   "I0": D.I0_NOISE, "alpha": D.ALPHA_NOISE, "n_views": D.N_VIEWS, "seed": a.seed},
                  open(meta_path, "w"), indent=1)
        print(f"calibrated A0 = {A0:.4f} on {os.path.basename(f0[0])}")

    print(f"{len(jobs)} slices from {a.patients} | levels={a.levels} | residual={a.residual} | {a.workers} workers")
    t, done = time.time(), 0
    with Pool(a.workers, initializer=init, initargs=(a, spacing, A0)) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            done += r
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                el = time.time() - t
                print(f"  {k + 1}/{len(jobs)}  ({el / 60:.1f} min, ~{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} min left)")
    print(f"done: {done} new slices in {a.out_dir}")
