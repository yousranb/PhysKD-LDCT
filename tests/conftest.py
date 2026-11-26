import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def run(*args):
    """Run a repository entry point; fail with its output if it does not exit cleanly."""
    r = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"{' '.join(args)}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """Tiny random dataset with the real file naming: 64 x 64 slices, values in [0,1]."""
    d = tmp_path_factory.mktemp("toy")
    npy, syn, synv = d / "npy", d / "syn", d / "syn_val"
    for p in (npy, syn, synv):
        p.mkdir()
    g = np.random.default_rng(0)
    for patient, n in (("L067", 4), ("L333", 2), ("L506", 2)):
        for i in range(n):
            clean = np.clip(0.25 + 0.05 * g.standard_normal((64, 64)), 0, 1).astype(np.float32)
            np.save(npy / f"{patient}_{i}_target.npy", clean)
            np.save(npy / f"{patient}_{i}_input.npy", np.clip(clean + 0.02 * g.standard_normal(clean.shape), 0, 1).astype(np.float32))
            out = syn if patient == "L067" else synv if patient == "L333" else None
            if out is not None:
                np.save(out / f"{patient}_{i}_cleanref.npy", clean)
                for fam in ("noise", "blur", "streak"):
                    np.save(out / f"{patient}_{i}_{fam}L1_input.npy",
                            np.clip(clean + 0.03 * g.standard_normal(clean.shape), 0, 1).astype(np.float32))
    return d
