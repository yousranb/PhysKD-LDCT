"""End-to-end smoke test of the entry points on a tiny random dataset (CPU, a few minutes)."""
import csv
import os

import pytest
import torch

from conftest import ROOT, run

SMALL = ["--crop", "32", "--batch_size", "2", "--epochs", "1", "--workers", "0", "--train_patients", "L067",
         "--val_patients", "L333"]


@pytest.fixture(scope="module")
def runs(toy, tmp_path_factory):
    r = tmp_path_factory.mktemp("runs")
    npy = str(toy / "npy")
    run("train.py", "--config", "configs/small_baseline.yaml", "--npy_dir", npy, "--out", str(r / "base"), *SMALL)
    for fam in ("noise", "streak", "blur"):        # small teachers to keep the test fast
        run("train.py", "--config", f"configs/teacher_{fam}.yaml", "--npy_dir", npy, "--syn_dir", str(toy / "syn"),
            "--val_syn_dir", str(toy / "syn_val"), "--width", "16", "--out", str(r / f"t_{fam}"), *SMALL)
    teachers = [f"{f}=" + str(r / f"t_{f}" / "model_best.pkl") for f in ("noise", "streak", "blur")]
    for cfg in ("physkd_v1", "physkd_v2", "physkd_v3", "small_resaug"):
        run("train_physkd.py", "--config", f"configs/{cfg}.yaml", "--npy_dir", npy, "--kd_syn_dir", str(toy / "syn"),
            "--teachers", *teachers, "--out", str(r / cfg), *SMALL)
    return r


def test_checkpoints_have_the_expected_format(runs):
    for name, width in (("base", 16), ("physkd_v1", 16), ("small_resaug", 16)):
        ck = torch.load(runs / name / "model_best.pkl", map_location="cpu", weights_only=False)
        assert ck["width"] == width and ck["block"] == "mamba1" and "net_model" in ck
    ck = torch.load(runs / "physkd_v2" / "model_best.pkl", map_location="cpu", weights_only=False)
    assert ck["kd"]["lambda_pix"] == 0.0 and ck["kd"]["lambda_feat"] == 1.0 and ck["kd"]["fusion"] == "spatial"


def test_resume(runs, toy):
    out = run("train_physkd.py", "--config", "configs/physkd_v1.yaml", "--npy_dir", str(toy / "npy"),
              "--kd_syn_dir", str(toy / "syn"), "--out", str(runs / "physkd_v1"), *SMALL[:-6], "--epochs", "2",
              "--workers", "0", "--train_patients", "L067", "--val_patients", "L333",
              "--teachers", *[f"{f}=" + str(runs / f"t_{f}" / "model_best.pkl") for f in ("noise", "streak", "blur")])
    assert "resuming from epoch 2" in out


def test_evaluate_and_tables(runs, toy, tmp_path):
    pytest.importorskip("lpips"); pytest.importorskip("piq")
    ev = tmp_path / "eval"
    try:
        for s in (0, 1):
            run("evaluate.py", "--ckpt", str(runs / "physkd_v1" / "model_best.pkl"), "--npy_dir", str(toy / "npy"),
                "--test_patients", "L506", "--out", str(ev / f"physkd_v1_s{s}" / "L506_3mm_25.csv"),
                "--save_dir", str(tmp_path / "pred"))
    except AssertionError as e:
        if "download" in str(e).lower() or "urlopen" in str(e).lower():
            pytest.skip("perceptual metric weights unavailable")
        raise
    assert len(os.listdir(tmp_path / "pred")) == 2
    run("make_tables.py", "--eval_dir", str(ev), "--out", str(tmp_path / "tables"), "--runs", "physkd_v1",
        "--sets", "L506_3mm_25", "--seeds", "0", "1")
    rows = list(csv.DictReader(open(tmp_path / "tables" / "Table_L506_3mm_25.csv")))
    assert rows[0]["SD_type"] == "between_seeds" and rows[0]["Seeds"] == "0,1"


def test_tables_refuse_missing_runs(tmp_path):
    import subprocess, sys
    r = subprocess.run([sys.executable, "make_tables.py", "--eval_dir", str(tmp_path), "--out", str(tmp_path / "t"),
                        "--runs", "physkd_v1", "--sets", "L506_3mm_25"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "missing evaluation files" in r.stdout
