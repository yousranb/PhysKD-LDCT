"""Evaluate a checkpoint (train.py or train_physkd.py) on full 512 x 512 test slices.

Writes one CSV row per slice for the LDCT input and for the model (method,slice,psnr,ssim,rmse,vif,lpips,dists,nqm),
on the fixed scale (HU + 1024) / 4096, with z = mu_z at inference. --save_dir also exports the predictions as .npy.
No slice is skipped: a missing target or an empty test set stops the script.

  python evaluate.py --ckpt runs/physkd_v1_s0/model_best.pkl --npy_dir data/npy_3mm --test_patients L506 \
      --out results/eval/physkd_v1_s0/L506_3mm_25.csv --save_dir results/pred/physkd_v1_s0/L506_3mm_25
"""
import argparse
import os

import numpy as np
import torch

from physkd.data import TEST_PATIENTS, list_pairs, list_syn_pairs
from physkd.metrics import KEYS, all_metrics, self_check
from physkd.models import build


@torch.no_grad()
def evaluate(a, dev):
    if not os.path.exists(a.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {a.ckpt}")
    ck = torch.load(a.ckpt, map_location=dev)
    block = ck.get("block", "mamba1")                     # read from the checkpoint
    net = build(block, ck.get("width", 32)).to(dev).eval()
    net.load_state_dict(ck["net_model"])
    print(f"checkpoint block: {block} | width {ck.get('width', 32)}")
    self_check(dev)
    if a.eval_syn_dir:                                  # synthetic degraded test pairs (teachers, Table II(b))
        pairs = list_syn_pairs(a.eval_syn_dir, a.test_patients, a.syn_families)
        print(f"synthetic test: {a.syn_families} from {a.eval_syn_dir}")
    else:
        pairs = list_pairs(a.npy_dir, a.test_patients)
    inp_name = "Degraded input" if a.eval_syn_dir else "LDCT input"
    print(f"{len(pairs)} test slices from {a.test_patients} | full-size slices, z = mu_z")
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
    rows = []
    for k, (inp, tgt) in enumerate(pairs):
        q = np.load(inp).astype(np.float32)
        f = np.load(tgt).astype(np.float64)
        y = net(torch.from_numpy(q)[None, None].to(dev))[0].squeeze().double().cpu().numpy()
        name = os.path.basename(inp).replace("_input.npy", "")
        if a.save_dir:
            np.save(os.path.join(a.save_dir, f"{name}_{a.suffix}.npy"), y.astype(np.float32))
        for method, x in ((inp_name, q.astype(np.float64)), (a.method_name, y)):
            rows.append({"method": method, "slice": name, **all_metrics(f, x, dev)})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(pairs)}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("method,slice," + ",".join(KEYS) + "\n")
        for r in rows:
            fh.write(f"{r['method']},{r['slice']}," + ",".join(f"{r[k]:.6g}" for k in KEYS) + "\n")
    pct = {"ssim", "rmse"}
    print(f"\nScale: fixed [0,1] = (HU + 1024) / 4096 | n = {len(pairs)} slices | patients {a.test_patients}")
    print(f"  {'metric':7s}" + "".join(f"{m:>22s}" for m in (inp_name, a.method_name)))
    for key in KEYS:
        line = f"  {key:7s}"
        for m in (inp_name, a.method_name):
            v = np.array([r[key] for r in rows if r["method"] == m]) * (100 if key in pct else 1)
            line += f"{v.mean():>14.4f} ± {v.std():<6.4f}"
        print(line)
    print(f"\nPer-slice values: {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npy_dir", help="test pairs (<patient>_<i>_input/target.npy)")
    ap.add_argument("--test_patients", nargs="+", default=TEST_PATIENTS,
                    help="patient prefixes, e.g. L506 (25 %%) or L506_10 / L506_5 (simulated doses)")
    ap.add_argument("--out", required=True, help="per-slice metrics CSV")
    ap.add_argument("--save_dir", default=None, help="export the predictions here as .npy")
    ap.add_argument("--suffix", default="pred", help="file suffix of the exported predictions")
    ap.add_argument("--method_name", default="PhysKD-LDCT")
    ap.add_argument("--eval_syn_dir", default=None, help="synthetic degraded test pairs instead of --npy_dir")
    ap.add_argument("--syn_families", nargs="+", default=["noise", "blur", "streak"])
    a = ap.parse_args()
    if not a.eval_syn_dir and not a.npy_dir:
        ap.error("give --npy_dir (or --eval_syn_dir)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(a, dev)
