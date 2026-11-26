import json
import os

import pytest
import torch

from physkd.models import build, count_parameters

HERE = os.path.dirname(__file__)


@pytest.mark.parametrize("width,expected", [(32, 7_892_545), (16, 1_989_665)])
def test_parameter_count(width, expected):
    assert count_parameters(build("mamba1", width)) == expected


@pytest.mark.parametrize("width", [16, 32])
def test_state_dict_matches_original_implementation(width):
    """Same parameter names and shapes as the original vae_mamba.py: its checkpoints load with strict=True."""
    ref = json.load(open(os.path.join(HERE, "original_state_dict_shapes.json")))[str(width)]
    new = {k: list(v.shape) for k, v in build("mamba1", width).state_dict().items()}
    assert new == ref


def test_forward_shape_identity_init_and_deterministic_eval():
    torch.manual_seed(0)
    net = build("mamba1", 16).eval()
    x = torch.rand(2, 1, 64, 64)
    with torch.no_grad():
        y, mu, logvar = net(x)
        y2, _, _ = net(x)
    assert y.shape == x.shape and mu.shape == (2, 128, 4, 4)
    assert torch.equal(y, x)           # zero-initialized head: the untrained network is the identity
    assert torch.equal(y, y2)          # z = mu at inference


def test_unknown_block_rejected():
    with pytest.raises(ValueError):
        build("ss2d", 16)
