from __future__ import annotations

import torch

from causal_rl.metrics.mmd import mmd2


def test_mmd_convergence() -> None:
    torch.manual_seed(0)
    x = torch.randn(2000, 3)
    y = x + 0.2
    z = x.clone()
    val_diff = float(mmd2(x, y).item())
    val_same = float(mmd2(x, z).item())
    assert val_same <= val_diff
