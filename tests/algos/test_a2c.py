from __future__ import annotations

import torch

from causal_rl.algos.on_policy.a2c import A2C


def test_a2c_update_runs() -> None:
    algo = A2C(obs_dim=6, n_actions=8)
    obs = torch.randn(32, 6)
    action, _ = algo.select_action(obs)
    batch = {
        "obs": obs,
        "action": action,
        "returns": torch.randn(32),
        "adv": torch.randn(32),
    }
    metrics = algo.update(batch)
    assert "value_loss" in metrics
