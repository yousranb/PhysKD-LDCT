"""Aggregate per-slice evaluation CSVs (evaluate.py) into the result tables of the paper.

Layout expected (scripts/evaluate.sh writes it):  <eval_dir>/<run>_s<seed>/<set>.csv
  run: full_baseline, small_baseline, small_resaug, physkd_v1, physkd_v2, physkd_v3, ...
  set: L506_3mm_25, L506_1mm_25, L506_3mm_10, L506_1mm_10, L506_3mm_5, L506_1mm_5, Mayo2020_25

Per run, a metric is averaged over test slices; for Mayo2020 it is averaged per patient first and then over patients
(--patient_level). Across runs (seeds) we report mean and SD (ddof = 1). With a single run, the SD is computed
between slices (ddof = 0) and SD_type says so. Missing CSVs are reported and stop the script (no silent skipping);
use --allow_missing to write the table with the available runs only (missing ones are listed in the output).

  python make_tables.py --eval_dir results/eval --out results/tables
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

METRICS = ["psnr", "ssim", "vif", "lpips", "dists", "nqm"]
RUNS = ["full_baseline", "small_baseline", "small_resaug", "physkd_v1", "physkd_v2", "physkd_v3"]
SETS = {"L506_3mm_25": ("3mm", "25%"), "L506_1mm_25": ("1mm", "25%"), "L506_3mm_10": ("3mm", "10%"),
        "L506_1mm_10": ("1mm", "10%"), "L506_3mm_5": ("3mm", "5%"), "L506_1mm_5": ("1mm", "5%"),
        "Mayo2020_25": ("", "Mayo2020")}


def read(path, method_rows=True):
    rows = [r for r in csv.DictReader(open(path)) if (r["method"] != "LDCT input") == method_rows]
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def run_mean(rows, patient_level):
    if not patient_level:
        return {k: float(np.mean([float(r[k]) for r in rows])) for k in METRICS}, len(rows)
    by = {}
    for r in rows:
        by.setdefault(r["slice"].rsplit("_", 1)[0], []).append(r)      # patient = slice name without the index
    return {k: float(np.mean([np.mean([float(r[k]) for r in v]) for v in by.values()])) for k in METRICS}, len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="results/eval")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--allow_missing", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    missing = []
    for st in a.sets:
        thick, dose = SETS[st]
        patient_level = st.startswith("Mayo2020")
        out_rows = []
        for run in a.runs:
            per, n, seeds = [], 0, []
            for s in a.seeds:
                f = os.path.join(a.eval_dir, f"{run}_s{s}", f"{st}.csv")
                if not os.path.exists(f):
                    missing.append(f); continue
                m, n = run_mean(read(f), patient_level); per.append(m); seeds.append(s)
                if len(per) == 1:
                    single_rows = read(f)
            if not per:
                continue
            row = {"Method": run, "Thickness": thick, "Dose": dose, "Seeds": ",".join(map(str, seeds)), "N": n}
            if len(per) > 1:
                row["SD_type"] = "between_seeds"
                for k in METRICS:
                    v = np.array([p[k] for p in per]); row[k.upper()], row[k.upper() + "_SD"] = v.mean(), v.std(ddof=1)
            else:
                row["SD_type"] = "between_slices"
                for k in METRICS:
                    v = np.array([float(r[k]) for r in single_rows])
                    row[k.upper()], row[k.upper() + "_SD"] = per[0][k], v.std()
            out_rows.append(row)
        if out_rows:
            path = os.path.join(a.out, f"Table_{st}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
            print(f"{path}: {len(out_rows)} methods")
    if missing:
        print("missing evaluation files:\n  " + "\n  ".join(missing))
        if not a.allow_missing:
            sys.exit(1)
