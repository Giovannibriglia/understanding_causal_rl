from __future__ import annotations

import torch

from causal_rl.algos.off_policy.dqn import DQN


def test_dqn_update_runs() -> None:
    algo = DQN(obs_dim=6, n_actions=8)
    batch = {
        "obs": torch.randn(32, 6),
        "action": torch.randint(0, 8, (32, 1)),
        "reward": torch.randn(32),
        "next_obs": torch.randn(32, 6),
        "done": torch.zeros(32),
    }
    metrics = algo.update(batch)
    assert metrics["td_loss"] >= 0.0
