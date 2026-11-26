"""Train the VAE-Mamba backbone without distillation: full/compact baselines, the synthetic-augmentation
baseline (VAE-Mamba-Aug), and the three degradation-specific teachers.

  baseline : real LDCT/FDCT pairs only                         (configs/full_baseline.yaml, small_baseline.yaml)
  teacher  : synthetic pairs of one degradation family only    (configs/teacher_{noise,streak,blur}.yaml)
             (--syn_ratio 1, validation on the same family of the validation patient)

Loss: lambda_rec * MSE(y_hat, y) + beta(t) * KL, beta linearly annealed to beta_max over beta_warmup epochs.
The checkpoint with the best validation PSNR is kept (model_best.pkl); model_last.pkl allows resuming.

  python train.py --config configs/small_baseline.yaml --npy_dir data/npy_3mm --out runs/small_baseline_s0 --seed 0
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from physkd import config
from physkd.data import (NpyPairs, TRAIN_PATIENTS, VAL_PATIENTS, create_loaders_npy, list_syn_pairs,
                         mix_synthetic)
from physkd.models import build, kl_loss


def psnr01(a, b):
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def train(a, dev):
    os.makedirs(a.out, exist_ok=True)
    tr, va = create_loaders_npy(a.npy_dir, a.train_patients, a.val_patients, a.crop, a.batch_size, a.workers)
    if a.syn_dir:
        tr = mix_synthetic(tr, a)
    if a.val_syn_dir:                                   # teachers: validate on their own degradation
        vpairs = list_syn_pairs(a.val_syn_dir, a.val_patients, a.syn_families)
        va = DataLoader(NpyPairs(vpairs, crop=a.crop, random_crop=False), batch_size=1, num_workers=a.workers)
        print(f"[val] {len(vpairs)} synthetic {a.syn_families} pairs of {a.val_patients} from {a.val_syn_dir}")
    net = build(a.block, a.width).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print(f"block: {a.block} | params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M | device {dev}")
    start, best = 0, -float("inf")
    last = os.path.join(a.out, "model_last.pkl")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        assert ck.get("block", "mamba1") == a.block, f"{last} was trained with --block {ck.get('block', 'mamba1')}"
        net.load_state_dict(ck["net_model"]); opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resuming from epoch {start + 1} (best val PSNR {best:.4f})")
    skipped = 0
    for ep in range(start, a.epochs):
        beta = a.beta_max * min(1.0, (ep + 1) / max(1, a.beta_warmup))
        net.train(); s_rec = s_kl = n = 0
        for i, (q, f) in enumerate(tr):
            q, f = q.to(dev), f.to(dev)
            y, mu, logvar = net(q)
            rec, kl = F.mse_loss(y, f), kl_loss(mu, logvar)
            loss = a.lambda_rec * rec + beta * kl
            if not torch.isfinite(loss):
                skipped += 1; opt.zero_grad(set_to_none=True)
                print(f"[guard] non-finite loss skipped ({skipped})")
                if skipped > 50:
                    raise RuntimeError("training diverged")
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            s_rec += rec.item(); s_kl += kl.item(); n += 1
            if (i + 1) % 50 == 0:
                print(f"  ep {ep + 1} it {i + 1}/{len(tr)} rec {s_rec / n:.3e} kl {s_kl / n:.3e}")
        net.eval(); ps = []
        with torch.no_grad():
            for q, f in va:
                ps.append(psnr01(net(q.to(dev))[0], f.to(dev)))
        vp = float(np.mean(ps))
        print(f"epoch {ep + 1}/{a.epochs} | beta {beta:.2e} | train rec {s_rec / max(n, 1):.3e} "
              f"kl {s_kl / max(n, 1):.3e} | val PSNR {vp:.4f} dB")
        if vp > best:
            best = vp
            torch.save({"net_model": net.state_dict(), "epoch": ep, "val_psnr": best, "block": a.block, "width": a.width},
                       os.path.join(a.out, "model_best.pkl"))
            print(f"  new best model (epoch {ep + 1})")
        torch.save({"net_model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best,
                    "block": a.block, "width": a.width}, last)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy_dir", help="real pairs, 3 mm, 25 %% dose (prepare_mayo.py)")
    ap.add_argument("--out", help="run folder")
    ap.add_argument("--train_patients", nargs="+", default=TRAIN_PATIENTS)
    ap.add_argument("--val_patients", nargs="+", default=VAL_PATIENTS)
    ap.add_argument("--width", type=int, default=32, help="base channel width: 32 = full (7.89 M), 16 = compact (1.99 M)")
    ap.add_argument("--block", default="mamba1", help=argparse.SUPPRESS)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_rec", type=float, default=1.0)
    ap.add_argument("--beta_max", type=float, default=1e-6, help="KL weight after annealing")
    ap.add_argument("--beta_warmup", type=int, default=10, help="epochs of linear KL annealing")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--syn_dir", default=None, help="synthetic pairs from make_synthetic.py")
    ap.add_argument("--syn_ratio", type=float, default=0.5, help="fraction of synthetic samples per epoch (teachers: 1)")
    ap.add_argument("--syn_families", nargs="+", default=["noise", "blur", "streak"])
    ap.add_argument("--val_syn_dir", default=None, help="validate on synthetic pairs (teachers) instead of real LDCT")
    ap.add_argument("--seed", type=int, default=0)
    a = config.parse(ap, required=("npy_dir", "out"))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(a, dev)
