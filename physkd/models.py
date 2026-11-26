"""Residual VAE-UNet with two Mamba selective state-space blocks at the bottleneck (PhysKD-LDCT backbone).

Pure PyTorch (no mamba-ssm / custom CUDA kernels), so it runs on any GPU or on CPU.

  encoder   : stem + 4 stages [residual block -> stride-2 conv], skips kept at every scale
  latent    : 1x1 convs -> mu_z, log sigma_z^2 ; z = mu + sigma * eps (training), z = mu (inference)
  bottleneck: 2 stacked Mamba blocks on the flattened latent tokens, x + Mamba(LayerNorm(x)),
              block 1 in raster order, block 2 in reversed raster order
  decoder   : 4 stages [upsample -> concat skip -> residual block], predicts r ; output = x_LD + r

build(width=32): 7.89 M parameters (full model and teachers); build(width=16): 1.99 M (compact student).
The module and parameter names are those of the original vae_mamba.py, so its checkpoints load unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- Mamba (pure PyTorch)
class Mamba(nn.Module):
    """Mamba (selective SSM) block, same parameters and maths as mamba_ssm.Mamba (v1),
    with the scan written as a plain loop over the sequence (fine for bottleneck lengths)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))      # inverse softplus
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):                                   # u: (B, L, d_model)
        Bsz, L, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)             # (B, L, Di) each
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :L]).transpose(1, 2)
        dt, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                   # (B, L, Di)
        A = -torch.exp(self.A_log.float())                  # (Di, N)
        dA = torch.exp(dt.unsqueeze(-1) * A)                # (B, L, Di, N)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
        h = u.new_zeros(Bsz, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D
        return self.out_proj(y * F.silu(z))


class MambaBlock(nn.Module):
    def __init__(self, dim, reverse=False):
        super().__init__()
        self.norm, self.mamba, self.reverse = nn.LayerNorm(dim), Mamba(dim), reverse

    def forward(self, x):                                   # x: (B, C, H, W)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                    # (B, HW, C)
        if self.reverse:
            t = t.flip(1)
        t = t + self.mamba(self.norm(t))
        if self.reverse:
            t = t.flip(1)
        return t.transpose(1, 2).reshape(B, C, H, W)



# ----------------------------------------------------------------------------- VAE-UNet
class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.n1 = nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout)
        self.c2, self.n2 = nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(h)) + self.skip(x))


def build(block="mamba1", width=32):
    """VAE-Mamba with base width w: channels (w, 2w, 4w, 8w), latent 8w (w=32 is the default model)."""
    return VAEMambaUNet(ch=(width, 2 * width, 4 * width, 8 * width), latent=8 * width, block=block)


class VAEMambaUNet(nn.Module):
    def __init__(self, ch=(32, 64, 128, 256), latent=256, n_mamba=2, block="mamba1"):
        super().__init__()
        self.stem = nn.Conv2d(1, ch[0], 3, padding=1)
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        cin = ch[0]
        for c in ch:                                        # 4 encoder stages
            self.enc.append(ResBlock(cin, c))
            self.down.append(nn.Conv2d(c, c, 4, stride=2, padding=1))
            cin = c
        self.to_mu = nn.Conv2d(ch[-1], latent, 1)
        self.to_logvar = nn.Conv2d(ch[-1], latent, 1)
        if block != "mamba1":
            raise ValueError(f"block '{block}' is not part of this release (only 'mamba1', the model of the paper)")
        self.mamba = nn.Sequential(*[MambaBlock(latent, reverse=(i % 2 == 1)) for i in range(n_mamba)])
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        cin = latent
        for c in reversed(ch):                              # 4 decoder stages
            self.up.append(nn.ConvTranspose2d(cin, c, 4, stride=2, padding=1))
            self.dec.append(ResBlock(2 * c, c))
            cin = c
        self.head = nn.Conv2d(ch[0], 1, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # starts as identity: y = x

    def forward(self, x):
        h, skips = self.stem(x), []
        for enc, down in zip(self.enc, self.down):
            h = enc(h); skips.append(h); h = down(h)
        mu, logvar = self.to_mu(h), self.to_logvar(h).clamp(-20, 10)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        h = self.mamba(z)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), s], 1))
        return x + self.head(h), mu, logvar


def kl_loss(mu, logvar):
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())
