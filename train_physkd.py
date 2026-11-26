"""Train a student with multi-teacher knowledge distillation (PhysKD-LDCT) or the matched ResAug control.

Frozen teachers (noise, streak, blur) guide a student trained on real LDCT/FDCT pairs. With --kd_syn_dir and
--kd_syn_extra, every step adds a batch of synthetic degraded inputs (training patients) on which the student
distils the fused teachers (no FDCT supervision there unless --kd_syn_rec).

  L = lambda_rec * ||y_S - y||^2 + lambda_kd * (lambda_pix * L_pix + lambda_feat * L_feat)        (real batch)
      + lambda_kd * (lambda_pix * L_pix + lambda_feat * L_feat)                                  (degraded batch)
      + beta(t) * (KL_real + KL_degraded) + ||y_T - y||^2                                         (fusion, teachers frozen)
  L_pix = ||y_S - y_T||^2 ; L_feat = ||A(mu_S) - z_T||^2 (A: 1x1 conv adapter); each KD term rescaled to the running
  magnitude of the reconstruction loss (EMA, 0.99), so lambda_pix and lambda_feat are relative weights.

  V1 output (1, 0) | V2 latent (0, 1) | V3 joint (0.5, 0.2)   -> configs/physkd_v{1,2,3}.yaml
  ResAug control: --lambda_kd 0 --kd_syn_rec (same batches, degraded inputs supervised by their reference)

The student checkpoint has the format of train.py (net_model, block, width): evaluate.py reads both.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from physkd import config
from physkd.data import NpyPairs, TRAIN_PATIENTS, VAL_PATIENTS, create_loaders_npy, list_syn_pairs
from physkd.distill import SpatialFusion, fuse, load_teacher, standardise, teachers_forward
from physkd.models import build, kl_loss


def psnr01(a, b):
    import math
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy_dir", help="real pairs, 3 mm, 25 %% dose (prepare_mayo.py)")
    ap.add_argument("--out", help="run folder")
    ap.add_argument("--teachers", nargs="+", default=None, help="name=checkpoint for the frozen teachers")
    ap.add_argument("--train_patients", nargs="+", default=TRAIN_PATIENTS)
    ap.add_argument("--val_patients", nargs="+", default=VAL_PATIENTS)
    ap.add_argument("--block", default="mamba1", help=argparse.SUPPRESS)
    ap.add_argument("--width", type=int, default=16, help="student base width (16 = compact student, 1.99 M)")
    ap.add_argument("--lambda_rec", type=float, default=1.0)
    ap.add_argument("--lambda_kd", type=float, default=1.0, help="0 = no distillation (ResAug control)")
    ap.add_argument("--lambda_pix", type=float, default=0.5, help="V1: 1, V2: 0, V3: 0.5")
    ap.add_argument("--lambda_feat", type=float, default=0.2, help="V1: 0, V2: 1, V3: 0.2")
    ap.add_argument("--fusion_target", choices=["gt", "kd"], default="gt")
    ap.add_argument("--feat_mode", choices=["mse", "norm"], default="mse")
    ap.add_argument("--kd_balance", choices=["ema", "none"], default="ema")
    ap.add_argument("--kd_syn_dir", default=None, help="synthetic degraded inputs (make_synthetic.py, training patients)")
    ap.add_argument("--kd_syn_extra", action="store_true", help="add --kd_syn_batch degraded inputs to the real batch")
    ap.add_argument("--kd_syn_batch", type=int, default=None, help="degraded inputs per step (default: batch_size)")
    ap.add_argument("--kd_syn_rec", action="store_true", help="also supervise the degraded inputs with their reference")
    ap.add_argument("--kd_syn_families", nargs="+", default=["noise", "blur", "streak"])
    ap.add_argument("--beta_max", type=float, default=1e-6)
    ap.add_argument("--beta_warmup", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = config.parse(ap, required=("npy_dir", "out", "teachers"))
    # options of the original script that the paper's runs left at their defaults
    # fixed settings of the paper's runs (kept in the checkpoint record, as in the original script)
    a.fusion, a.kd_schedule, a.kd_select, a.train_fraction, a.val_syn_dir = "spatial", "const", False, 1.0, None
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    spec = a.teachers
    names = [s.split("=", 1)[0] for s in spec]
    teachers = [load_teacher(s.split("=", 1)[1], dev) for s in spec]
    k = len(teachers)

    tr, va = create_loaders_npy(a.npy_dir, a.train_patients, a.val_patients, a.crop, a.batch_size, a.workers)
    steps = len(tr.dataset) // a.batch_size                 # same steps/epoch as the baseline and ResAug
    if a.kd_syn_dir:
        b_real = a.batch_size if a.kd_syn_extra else a.batch_size // 2
        b_syn = (a.kd_syn_batch or a.batch_size) if a.kd_syn_extra else a.batch_size - b_real
        tr = DataLoader(tr.dataset, batch_size=b_real, shuffle=True, num_workers=a.workers, drop_last=True)
        syn = NpyPairs(list_syn_pairs(a.kd_syn_dir, a.train_patients, a.kd_syn_families), crop=a.crop, random_crop=True)
        tr_syn = DataLoader(syn, batch_size=b_syn, shuffle=True, num_workers=a.workers, drop_last=True)
        print(f"[E6{' add' if a.kd_syn_extra else ''}] {len(syn)} degraded inputs from {a.kd_syn_dir} | batch = {b_real} real + {b_syn} "
              f"degraded | degraded target: {'FDCT + teachers' if a.kd_syn_rec else 'teachers only'} | {steps} steps/epoch")
    va_syn = None
    select = "real"                                        # model selection on real validation pairs (L333)


    def forever(loader):
        while True:
            for b in loader:
                yield b

    student = build(a.block, a.width).to(dev)
    fusion = SpatialFusion(k).to(dev)
    ls, lt = student.to_mu.out_channels, teachers[0].to_mu.out_channels
    use_adapter = a.feat_mode == "norm" or ls != lt
    adapter = nn.Conv2d(ls, lt, 1).to(dev)
    if ls == lt:
        with torch.no_grad():
            adapter.weight.copy_(torch.eye(ls).view(ls, ls, 1, 1)); adapter.bias.zero_()   # identity start
    params = list(student.parameters()) + list(fusion.parameters()) + (list(adapter.parameters()) if use_adapter else [])
    print(f"student width {a.width}: {sum(p.numel() for p in student.parameters()) / 1e6:.2f} M params "
          f"(teacher {sum(p.numel() for p in teachers[0].parameters()) / 1e6:.2f} M) | latent {ls} -> {lt}")
    opt = torch.optim.Adam(params, lr=a.lr)
    cfg = {k_: getattr(a, k_) for k_ in ("lambda_rec", "lambda_kd", "lambda_pix", "lambda_feat", "fusion", "fusion_target",
                                         "feat_mode", "kd_schedule", "kd_select", "kd_balance", "train_fraction", "beta_max",
                                         "lr", "seed", "kd_syn_dir", "kd_syn_rec", "kd_syn_families", "val_syn_dir", "kd_syn_extra", "width")}
    cfg["teachers"] = spec
    print(f"teachers {names} | student {a.block} | " + " ".join(f"{k_} {v}" for k_, v in cfg.items() if k_ != "teachers")
          + f" | device {dev}")

    start, best = 0, -float("inf")
    last = os.path.join(a.out, "model_last.pkl")
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        assert ck.get("kd", {}).get("teachers") == spec, "model_last.pkl was trained with other teachers"
        student.load_state_dict(ck["net_model"]); fusion.load_state_dict(ck["fusion"])
        adapter.load_state_dict(ck["adapter"]); opt.load_state_dict(ck["opt"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resuming from epoch {start + 1} (best val PSNR {best:.4f})")

    skipped = 0
    ema = {}

    def balanced(name, term, ref):
        """term rescaled to the running magnitude of ref (L_rec); gradients keep the term's direction."""
        if a.kd_balance == "none":
            return term
        for k_, v in ((name, term), ("rec", ref)):
            v = float(v.detach())
            ema[k_] = v if k_ not in ema else 0.99 * ema[k_] + 0.01 * v
        return term * (ema["rec"] / max(ema[name], 1e-12))

    for ep in range(start, a.epochs):
        beta = a.beta_max * min(1.0, (ep + 1) / max(1, a.beta_warmup))
        lkd = a.lambda_kd                                     # constant distillation weight
        student.train(); fusion.train()
        acc = dict(rec=0.0, pix=0.0, feat=0.0, fuse=0.0, kl=0.0); n = 0
        if a.kd_syn_dir:
            acc.update(pix_deg=0.0, feat_deg=0.0, rec_deg=0.0)
        it_real = forever(tr)
        it_syn = forever(tr_syn) if a.kd_syn_dir else None
        for i in range(steps):
            q, f = next(it_real)
            q, f = q.to(dev), f.to(dev)
            ys, mus = teachers_forward(teachers, q)
            w = fusion(q, ys, mus)
            y_t, z_t = fuse(w, ys, mus, a.feat_mode)
            y_s, mu_s, lv_s = student(q)
            rec = F.mse_loss(y_s, f)
            gt_mode = a.fusion_target == "gt"
            fuse_loss = F.mse_loss(y_t, f) if gt_mode else torch.zeros((), device=dev)
            yt_kd, zt_kd = (y_t.detach(), z_t.detach()) if a.fusion_target == "gt" else (y_t, z_t)
            pix = F.mse_loss(y_s, yt_kd)
            zs = adapter(mu_s) if use_adapter else mu_s
            zs = standardise(zs) if a.feat_mode == "norm" else zs
            feat = F.mse_loss(zs, zt_kd)
            kl = kl_loss(mu_s, lv_s)
            kd = (a.lambda_pix * balanced("pix", pix, rec) if a.lambda_pix else 0.0) \
                + (a.lambda_feat * balanced("feat", feat, rec) if a.lambda_feat else 0.0)
            loss = a.lambda_rec * rec + lkd * kd + beta * kl + fuse_loss
            extra = {}
            if a.kd_syn_dir:                                  # E6: distil on degraded inputs
                qd, fd = next(it_syn)
                qd, fd = qd.to(dev), fd.to(dev)
                ysd, musd = teachers_forward(teachers, qd)
                wd = fusion(qd, ysd, musd)
                ytd, ztd = fuse(wd, ysd, musd, a.feat_mode)
                y_sd, mu_sd, lv_sd = student(qd)
                pix_d = F.mse_loss(y_sd, ytd.detach())
                zd = adapter(mu_sd) if use_adapter else mu_sd
                feat_d = F.mse_loss(standardise(zd) if a.feat_mode == "norm" else zd, ztd.detach())
                kd_d = (a.lambda_pix * balanced("pix_deg", pix_d, rec) if a.lambda_pix else 0.0) \
                    + (a.lambda_feat * balanced("feat_deg", feat_d, rec) if a.lambda_feat else 0.0)
                loss = loss + lkd * kd_d + beta * kl_loss(mu_sd, lv_sd)
                extra = {"pix_deg": pix_d, "feat_deg": feat_d}
                if a.kd_syn_rec:                              # ResAug + KD: degraded half also sees its target
                    rec_d = F.mse_loss(y_sd, fd)
                    loss = loss + a.lambda_rec * rec_d + (F.mse_loss(ytd, fd) if gt_mode else 0.0)
                    extra["rec_deg"] = rec_d
            if not torch.isfinite(loss):
                skipped += 1; opt.zero_grad(set_to_none=True)
                print(f"[guard] non-finite loss skipped ({skipped})")
                if skipped > 50:
                    raise RuntimeError("training diverged")
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            for k_, v in (("rec", rec), ("pix", pix), ("feat", feat), ("fuse", fuse_loss), ("kl", kl), *extra.items()):
                acc[k_] += float(v)
            n += 1
            if (i + 1) % 50 == 0:
                print(f"  ep {ep + 1} it {i + 1}/{steps} " + " ".join(f"{k_} {v / n:.3e}" for k_, v in acc.items()))

        student.eval(); fusion.eval(); ps, pt, ws = [], [], []
        with torch.no_grad():
            for q, f in va:
                q, f = q.to(dev), f.to(dev)
                ys, mus = teachers_forward(teachers, q)
                w = fusion(q, ys, mus); y_t, _ = fuse(w, ys, mus, a.feat_mode)
                ps.append(psnr01(student(q)[0], f)); pt.append(psnr01(y_t, f)); ws.append(w.mean(dim=(2, 3)).cpu())
            vs = [psnr01(student(q.to(dev))[0], f.to(dev)) for q, f in va_syn] if va_syn is not None else []
        vreal, wm = float(np.mean(ps)), torch.cat(ws).mean(0).tolist()
        vsyn = float(np.mean(vs)) if vs else float("nan")
        vp = {"real": vreal, "syn": vsyn, "mean": (vreal + vsyn) / 2}[select]
        print(f"epoch {ep + 1}/{a.epochs} | beta {beta:.2e} lambda_kd {lkd:.3f} | "
              + " ".join(f"{k_} {v / max(n, 1):.3e}" for k_, v in acc.items())
              + f" | val PSNR student {vreal:.4f}" + (f" degraded {vsyn:.4f}" if vs else "")
              + f" fused-teachers {np.mean(pt):.4f} | select[{select}] {vp:.4f} | w "
              + " ".join(f"{nm} {x:.3f}" for nm, x in zip(names, wm)))
        if vp > best:
            best = vp
            torch.save({"net_model": student.state_dict(), "block": a.block, "width": a.width, "epoch": ep, "val_psnr": best,
                        "fusion": fusion.state_dict(), "fusion_weights_val": dict(zip(names, wm)), "kd": cfg},
                       os.path.join(a.out, "model_best.pkl"))
            print(f"  new best model (epoch {ep + 1})")
        torch.save({"net_model": student.state_dict(), "fusion": fusion.state_dict(), "adapter": adapter.state_dict(),
                    "opt": opt.state_dict(), "epoch": ep, "best": best, "block": a.block, "width": a.width, "kd": cfg}, last)


if __name__ == "__main__":
    main()
