import glob
import os

import yaml

from physkd.data import TEST_PATIENTS, TRAIN_PATIENTS, VAL_PATIENTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_split_is_disjoint():
    assert TRAIN_PATIENTS == ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"]
    assert VAL_PATIENTS == ["L333"] and TEST_PATIENTS == ["L506"]
    assert not set(TRAIN_PATIENTS) & set(VAL_PATIENTS + TEST_PATIENTS)


def test_configs_do_not_change_the_split_and_match_the_paper():
    for f in glob.glob(os.path.join(ROOT, "configs", "*.yaml")):
        c = yaml.safe_load(open(f))
        assert "train_patients" not in c and "val_patients" not in c, f
        assert c["lr"] == 1e-4 and c["batch_size"] == 8 and c["crop"] == 256 and c["epochs"] == 100, f
        assert c["beta_max"] == 1e-6 and c["beta_warmup"] == 10, f
    weights = {v: yaml.safe_load(open(os.path.join(ROOT, "configs", f"physkd_v{v}.yaml"))) for v in (1, 2, 3)}
    assert [(weights[v]["lambda_pix"], weights[v]["lambda_feat"]) for v in (1, 2, 3)] == [(1.0, 0.0), (0.0, 1.0), (0.5, 0.2)]
    assert all(weights[v]["width"] == 16 and weights[v]["kd_syn_extra"] and not weights[v]["kd_syn_rec"] for v in (1, 2, 3))
    resaug = yaml.safe_load(open(os.path.join(ROOT, "configs", "small_resaug.yaml")))
    assert resaug["lambda_kd"] == 0.0 and resaug["kd_syn_rec"] and resaug["width"] == 16
