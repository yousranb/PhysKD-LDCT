import csv
import os

from conftest import run

ROWS = [("C1_0", 30.0), ("C1_1", 30.0), ("C1_2", 30.0), ("C2_0", 20.0)]


def write(path, psnr_by_slice):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("method,slice,psnr,ssim,rmse,vif,lpips,dists,nqm\n")
        for s, p in psnr_by_slice:
            fh.write(f"LDCT input,{s},1,1,1,1,1,1,1\nM,{s},{p},0.9,0.01,0.5,0.1,0.1,30\n")


def test_mayo2020_is_averaged_per_patient_then_over_seeds(tmp_path):
    for s in (0, 1, 2):
        write(str(tmp_path / f"physkd_v2_s{s}" / "Mayo2020_25.csv"), [(n, p + s) for n, p in ROWS])
    run("make_tables.py", "--eval_dir", str(tmp_path), "--out", str(tmp_path / "t"), "--runs", "physkd_v2",
        "--sets", "Mayo2020_25")
    r = list(csv.DictReader(open(tmp_path / "t" / "Table_Mayo2020_25.csv")))[0]
    # patient means 30+s and 20+s -> 25+s per seed -> mean 26, SD 1 (a slice average would give 27.5+s)
    assert float(r["PSNR"]) == 26.0 and abs(float(r["PSNR_SD"]) - 1.0) < 1e-9
