from __future__ import annotations

import torch

from causal_rl.algos.off_policy.ddpg import DDPG


def test_ddpg_update_runs() -> None:
    algo = DDPG(obs_dim=4, act_dim=3)
    batch = {
        "obs": torch.randn(32, 4),
        "action": torch.rand(32, 3),
        "reward": torch.randn(32),
        "next_obs": torch.randn(32, 4),
        "done": torch.zeros(32),
    }
    metrics = algo.update(batch)
    assert "td_loss" in metrics
