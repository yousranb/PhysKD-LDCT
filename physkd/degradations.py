"""Physics-motivated degradation operators used to build the teacher training tasks (projection domain).

Noise: Poisson counts with a scaled noise residual (gamma); blur: Gaussian smoothing of the sinogram along the
detector (sigma, detector bins); streaks: smooth angularly localized projection error field (delta_L), amplitude
calibrated once (calibrate_A0). Parallel-beam Radon transform with N_VIEWS views over 180 degrees, ramp-filtered FBP.
These operators define training tasks; they are not the 10 %/5 % dose simulations used for zero-shot testing.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.transform import radon, iradon

# ============================================================
# CONSTANTS
# ============================================================
MU_WATER = 0.20
N_VIEWS = 720

I0_NOISE = 1e7
ALPHA_NOISE = 0.10

NOISE_LEVELS = {
    "L0": {"gamma": 1.0},
    "L1": {"gamma": 2.0},
    "L2": {"gamma": 2.5},
    "L3": {"gamma": 4.0},
}

BLUR_LEVELS = {
    "L0": {"sigma": 0.60},
    "L1": {"sigma": 1.00},
    "L2": {"sigma": 1.60},
    "L3": {"sigma": 2.40},
}

STREAK_LEVELS = {
    "L0": {"delta_L": 0.25},
    "L1": {"delta_L": 0.50},
    "L2": {"delta_L": 1.00},
    "L3": {"delta_L": 2.00},
}

# ============================================================
# BASIC CT HELPERS
# ============================================================
def hu2mu(x):
    return MU_WATER * (1 + x / 1000.0)

def mu2hu(x):
    return 1000.0 * (x / MU_WATER - 1)

def reconstruct(p, theta, ds_cm, n):
    r = iradon(p, theta=theta, filter_name="ramp", circle=False, output_size=n).astype(np.float32)
    return mu2hu(r / ds_cm).astype(np.float32)

def clean_reference(hu, ds_cm):
    """
    hu    : 2D CT slice in HU
    ds_cm : pixel spacing in cm
    """
    mu = np.maximum(hu2mu(np.clip(hu, -1000, 3000)), 0).astype(np.float32)
    theta = np.linspace(0, 180, N_VIEWS, endpoint=False)
    p = radon(mu, theta=theta, circle=False).astype(np.float32) * ds_cm
    hu_ref = reconstruct(p, theta, ds_cm, hu.shape[0])
    return {"mu": mu, "theta": theta, "p": p, "hu": hu_ref}

# ============================================================
# 1) NOISE
# ============================================================
def sim_noise(c, ds_cm, gamma, seed, I0=I0_NOISE, alpha=ALPHA_NOISE):
    """
    Signal-dependent Poisson noise in projection domain.
    gamma controls severity.
    """
    rng = np.random.default_rng(seed)
    incident = alpha * I0
    lam = np.clip(incident * np.exp(-c["p"]), 1e-8, incident)
    k = rng.poisson(lam)

    p_phys = (-np.log(np.clip((k + 0.5) / incident, 1e-12, None))).astype(np.float32)
    residual = p_phys - c["p"]
    p_deg = (c["p"] + gamma * residual).astype(np.float32)

    return {
        "hu": reconstruct(p_deg, c["theta"], ds_cm, c["mu"].shape[0]),
        "p": p_deg,
        "p_phys": p_phys,
        "noise_residual": residual,
        "gamma": float(gamma),
    }

# ============================================================
# 2) BLUR
# ============================================================
def sim_blur(c, ds_cm, sigma):
    """
    Blur in projection domain (detector-direction Gaussian smoothing).
    """
    p_deg = gaussian_filter1d(c["p"], sigma=sigma, axis=0, mode="nearest").astype(np.float32)
    return {
        "hu": reconstruct(p_deg, c["theta"], ds_cm, c["mu"].shape[0]),
        "p": p_deg,
        "sigma": float(sigma),
    }

# ============================================================
# 3) STREAKS
# ============================================================
def raised_cosine_window(theta, theta0=25.0, theta1=65.0, taper_deg=8.0):
    theta = np.asarray(theta, dtype=np.float32)

    if not (0.0 <= theta0 < theta1 <= 180.0):
        raise ValueError("Expected 0 <= theta0 < theta1 <= 180.")

    width = theta1 - theta0
    taper_deg = float(np.clip(taper_deg, 1e-3, width / 2.0))

    w = np.zeros_like(theta, dtype=np.float32)
    inside = (theta >= theta0) & (theta <= theta1)

    plateau = inside & (theta >= theta0 + taper_deg) & (theta <= theta1 - taper_deg)
    w[plateau] = 1.0

    rise = inside & (theta >= theta0) & (theta < theta0 + taper_deg)
    x = (theta[rise] - theta0) / taper_deg
    w[rise] = 0.5 * (1.0 - np.cos(np.pi * x))

    fall = inside & (theta > theta1 - taper_deg) & (theta <= theta1)
    x = (theta1 - theta[fall]) / taper_deg
    w[fall] = 0.5 * (1.0 - np.cos(np.pi * x))

    w[~inside] = 0.0
    return w.astype(np.float32)

def smooth_detector_window(path_length, theta, theta0=25.0, theta1=65.0,
                           top_percent=12.0, path_gamma=1.7,
                           sigmoid_width=0.05, smooth_sigma_detector=2.0):
    eps = 1e-8
    L = np.asarray(path_length, dtype=np.float32)

    Ln = L / (np.max(L) + eps)
    score = Ln ** float(path_gamma)

    theta = np.asarray(theta, dtype=np.float32)
    in_sector = (theta >= theta0) & (theta <= theta1)

    if not np.any(in_sector):
        raise ValueError("No projection angles lie inside the selected sector.")

    sector_values = score[:, in_sector]
    valid_values = sector_values[sector_values > 0]

    if valid_values.size == 0:
        raise RuntimeError("No non-zero body paths were found inside the sector.")

    tau = np.percentile(valid_values, 100.0 - float(top_percent))

    beta = max(float(sigmoid_width), 1e-4)
    z = np.clip((score - tau) / beta, -40.0, 40.0)
    ws = 1.0 / (1.0 + np.exp(-z))

    ws[:, ~in_sector] = 0.0

    if smooth_sigma_detector > 0:
        ws = gaussian_filter1d(ws, sigma=float(smooth_sigma_detector), axis=0, mode="nearest")

    ws[:, ~in_sector] = 0.0
    ws /= np.max(ws) + eps

    return ws.astype(np.float32), float(tau)

def build_projection_error_field(c, ds_cm, theta0=25.0, theta1=65.0,
                                 angular_taper_deg=8.0, top_percent=12.0,
                                 path_gamma=1.7, sigmoid_width=0.05,
                                 smooth_sigma_detector=2.0):
    """
    Build a fixed smooth projection error field e0(s, theta).
    """
    eps = 1e-8

    m = (c["mu"] > 1e-6).astype(np.float32)
    L = radon(m, theta=c["theta"], circle=False).astype(np.float32)
    L *= float(ds_cm)

    wtheta = raised_cosine_window(c["theta"], theta0=theta0, theta1=theta1,
                                  taper_deg=angular_taper_deg)

    ws, tau = smooth_detector_window(
        L, c["theta"],
        theta0=theta0, theta1=theta1,
        top_percent=top_percent,
        path_gamma=path_gamma,
        sigmoid_width=sigmoid_width,
        smooth_sigma_detector=smooth_sigma_detector
    )

    e0 = (ws * wtheta[None, :]).astype(np.float32)

    outside = (c["theta"] < theta0) | (c["theta"] > theta1)
    e0[:, outside] = 0.0

    e0 /= np.max(np.abs(e0)) + eps

    return {
        "e0": e0,
        "tau": tau,
        "theta0": float(theta0),
        "theta1": float(theta1),
    }

def calibrate_A0(c, ds_cm, streak_model, calibration_delta_L=2.0, target_p95_hu=120.0):
    """
    Calibrate one global amplitude A0 once.
    """
    e0 = streak_model["e0"]
    p_test = (c["p"] + float(calibration_delta_L) * e0).astype(np.float32)
    hu_test = reconstruct(p_test, c["theta"], ds_cm, c["mu"].shape[0])
    residual = (hu_test - c["hu"]).astype(np.float32)

    p95_unit = float(np.percentile(np.abs(residual), 95.0))
    if p95_unit < 1e-8:
        raise RuntimeError("Calibration failed: negligible image-domain residual.")

    A0 = float(target_p95_hu) / p95_unit
    return float(A0)

def sim_streak(c, ds_cm, delta_L, streak_model, A0):
    """
    Apply streaks:
        p_deg = p_clean + A0 * delta_L * e0
    """
    e0 = streak_model["e0"]
    correction = (float(A0) * float(delta_L) * e0).astype(np.float32)
    p_deg = (c["p"] + correction).astype(np.float32)
    hu_deg = reconstruct(p_deg, c["theta"], ds_cm, c["mu"].shape[0])

    return {
        "hu": hu_deg,
        "p": p_deg,
        "corr": correction,
        "delta_L": float(delta_L),
        "A0": float(A0),
    }