from __future__ import annotations

import gymnasium as gym
import torch

from causal_rl.algos.on_policy.ppo import PPO


def test_ppo_cartpole_canary() -> None:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    algo = PPO(obs_dim=4, n_actions=2)
    action, log_prob = algo.select_action(obs_t)
    assert action.shape == (1, 1)
    batch = {
        "obs": obs_t.repeat(8, 1),
        "action": action.repeat(8, 1),
        "log_prob": log_prob.repeat(8),
        "adv": torch.randn(8),
        "returns": torch.randn(8),
        "value": torch.zeros(8),
    }
    metrics = algo.update(batch)
    assert "policy_loss" in metrics
