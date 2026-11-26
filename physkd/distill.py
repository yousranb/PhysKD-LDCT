"""Multi-teacher distillation components (PhysKD-LDCT, stage 2).

Frozen teachers (load_teacher), adaptive spatial fusion g_psi (SpatialFusion: per-pixel softmax weights from the
input and the teacher outputs; the same maps, average-pooled to the latent grid, combine the teacher latents),
fused targets y_T = sum_i w_i * y_i and z_T = sum_i w_i * z_i (fuse).
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from physkd.models import build


def load_teacher(path, dev):
    if not os.path.exists(path):
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")
    ck = torch.load(path, map_location=dev)
    net = build(ck.get("block", "mamba1"), ck.get("width", 32)).to(dev).eval()
    net.load_state_dict(ck["net_model"])
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class SpatialFusion(nn.Module):
    """Per-pixel weight maps from [LDCT input, teacher outputs]: each teacher can act where it is competent."""

    def __init__(self, k, width=32):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Conv2d(k + 1, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, k, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)          # start equal

    def forward(self, q, ys, mus):
        return torch.softmax(self.net(torch.cat([q] + ys, 1)), 1)                       # (B, k, H, W)


def standardise(z, eps=1e-5):
    """Per-sample, per-channel z-score over the spatial grid."""
    m = z.mean(dim=(2, 3), keepdim=True)
    s = z.std(dim=(2, 3), keepdim=True)
    return (z - m) / (s + eps)


def fuse(w, ys, mus, feat_mode):
    k = len(ys)
    y_t = sum(w[:, i:i + 1] * ys[i] for i in range(k))
    wl = w if w.shape[-1] == 1 else F.adaptive_avg_pool2d(w, mus[0].shape[-2:])      # weights on the latent grid
    zs = [standardise(m) for m in mus] if feat_mode == "norm" else mus
    z_t = sum(wl[:, i:i + 1] * zs[i] for i in range(k))
    return y_t, z_t


@torch.no_grad()
def teachers_forward(teachers, q):
    ys, mus = [], []
    for t in teachers:
        y, mu, _ = t(q)
        ys.append(y); mus.append(mu)
    return ys, mus
