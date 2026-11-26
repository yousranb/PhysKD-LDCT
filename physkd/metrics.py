"""Image-quality metrics on the fixed [0,1] scale, x = (HU + 1024) / 4096 (data_range = 1, no windowing).

PSNR, SSIM (scikit-image), VIF (piq.vif_p), LPIPS (AlexNet, lpips package), DISTS (piq), NQM (physkd/nqm.py, the
implementation used for every method of the paper). Both images are clipped to [0,1] before scoring.
First use downloads the AlexNet / VGG16 / DISTS weights to ~/.cache/torch.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

KEYS = ["psnr", "ssim", "rmse", "vif", "lpips", "dists", "nqm"]
_nets = {}


def _get(dev):
    if dev not in _nets:
        import lpips
        import piq
        _nets[dev] = (lpips.LPIPS(net="alex", verbose=False).to(dev).eval(),
                      piq.DISTS(reduction="none").to(dev).eval())
    return _nets[dev]


def compute_nqm(img_ref, img_dist):
    """NQM (dB, higher is better) from physkd/nqm.py, the implementation used for all methods."""
    from physkd.nqm import nqm as _nqm_user
    return _nqm_user(np.asarray(img_ref, np.float32), np.asarray(img_dist, np.float32))


@torch.no_grad()
def all_metrics(ref, x, dev):
    """ref, x: 2-D arrays on the fixed [0,1] scale (clipped to [0,1] before scoring)."""
    import piq
    lp, di = _get(dev)
    ref = np.clip(np.asarray(ref, np.float64), 0, 1)
    x = np.clip(np.asarray(x, np.float64), 0, 1)
    mse = np.mean((ref - x) ** 2)
    r = torch.from_numpy(ref).float()[None, None].to(dev)
    t = torch.from_numpy(x).float()[None, None].to(dev)
    r3, t3 = r.repeat(1, 3, 1, 1), t.repeat(1, 3, 1, 1)
    return {
        "psnr": float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse)),
        "ssim": float(ssim(ref, x, data_range=1.0)),
        "rmse": float(np.sqrt(mse)),
        "vif": piq.vif_p(t, r, data_range=1.0).item(),
        "lpips": lp(t3, r3, normalize=True).item(),
        "dists": di(t3, r3).item(),
        "nqm": float(compute_nqm(ref, x)),
    }


def self_check(dev):
    """Identical images must give LPIPS 0, DISTS 0, VIF 1, NQM inf."""
    g = np.random.default_rng(0)
    img = gaussian_filter(g.random((256, 256)), 3)
    img = (img - img.min()) / (img.max() - img.min())
    chk = all_metrics(img, img, dev)
    print("self-check (identical images):", {k: round(v, 4) for k, v in chk.items()})
    if abs(chk["lpips"]) > 1e-3 or abs(chk["dists"]) > 1e-3 or abs(chk["vif"] - 1) > 1e-3:
        print("WARNING: perceptual metrics fail the identity check - check the downloaded weights")
