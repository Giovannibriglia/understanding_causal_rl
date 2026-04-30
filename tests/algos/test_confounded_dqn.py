from __future__ import annotations

import torch

from causal_rl.algos.off_policy.confounded_dqn import ConfoundedDQN


def test_confounded_dqn_penalty_applied() -> None:
    algo = ConfoundedDQN(obs_dim=6, n_actions=8, sensitivity_lambda=1.0)
    batch = {
        "obs": torch.randn(32, 6),
        "action": torch.randint(0, 8, (32, 1)),
        "reward": torch.randn(32),
        "next_obs": torch.randn(32, 6),
        "done": torch.zeros(32),
        "delta_tv": torch.tensor(0.2),
    }
    metrics = algo.update(batch)
    assert metrics["policy_loss"] >= 0.0
