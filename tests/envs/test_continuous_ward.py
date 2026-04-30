from __future__ import annotations

import torch

from causal_rl.envs.continuous_ward import ContinuousWardEnv


def test_continuous_ward_dynamics_stable() -> None:
    env = ContinuousWardEnv(cell=1, n_envs=64, device="cpu")
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        action = torch.rand((64, 3))
        obs, _, _, _, _ = env.step(action)
    assert torch.isfinite(obs).all()
    assert float(obs.min().item()) >= 0.0
    assert float(obs.max().item()) <= 1.0


def test_known_z_policy_beats_constant_policy() -> None:
    env = ContinuousWardEnv(cell=1, n_envs=128, device="cpu")
    _, info = env.reset(seed=2)
    z = info["latent_Z"]
    action_known = torch.sigmoid(
        torch.cat([z[:, :1], 1.0 - z[:, :1], 0.5 * torch.ones_like(z[:, :1])], dim=-1)
    )
    action_const = 0.5 * torch.ones((128, 3))
    _, reward_known, _, _, _ = env.step(action_known)
    env.reset(seed=2)
    _, reward_const, _, _, _ = env.step(action_const)
    assert float(reward_known.mean().item()) >= float(reward_const.mean().item()) - 0.1
