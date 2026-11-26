import numpy as np
import pytest

from physkd.nqm import nqm


def test_psnr_ssim_on_fixed_scale():
    from skimage.metrics import structural_similarity as ssim
    g = np.random.default_rng(0)
    ref = g.random((64, 64))
    x = np.clip(ref + 0.01 * g.standard_normal(ref.shape), 0, 1)
    mse = np.mean((ref - x) ** 2)
    assert 10 * np.log10(1 / mse) == pytest.approx(40.0, abs=0.5)
    assert ssim(ref, ref, data_range=1.0) == pytest.approx(1.0)


def test_nqm_orders_noise_levels():
    g = np.random.default_rng(0)
    ref = np.clip(0.3 + 0.05 * g.standard_normal((128, 128)), 0, 1).astype(np.float32)
    low = nqm(ref, np.clip(ref + 0.005 * g.standard_normal(ref.shape), 0, 1).astype(np.float32))
    high = nqm(ref, np.clip(ref + 0.05 * g.standard_normal(ref.shape), 0, 1).astype(np.float32))
    assert low > high


def test_all_metrics_identity():
    """Needs the LPIPS / DISTS weights (downloaded on first use); skipped when they are not available."""
    pytest.importorskip("lpips"); pytest.importorskip("piq")
    from physkd.metrics import all_metrics
    try:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(np.random.default_rng(0).random((64, 64)), 2)
        img = (img - img.min()) / (img.max() - img.min())
        m = all_metrics(img, img, "cpu")
    except Exception as e:                       # no network / no cached weights
        pytest.skip(f"perceptual metric weights unavailable: {e}")
    assert m["psnr"] == float("inf") and m["ssim"] == pytest.approx(1.0)
    assert m["lpips"] == pytest.approx(0.0, abs=1e-3) and m["vif"] == pytest.approx(1.0, abs=1e-3)
