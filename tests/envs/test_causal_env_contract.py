from __future__ import annotations

import torch

from causal_rl.envs.registry import ENVS, make


def test_registered_envs_satisfy_contract() -> None:
    for name in ENVS:
        env = make(name, cell=1, n_envs=8, device="cpu")
        obs, info = env.reset(seed=0)
        assert "latent_Z" in info
        action = torch.randint(0, 8, (8, 1)) if env.is_discrete_action else torch.rand((8, 3))
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs.shape[0] == 8
        assert reward.shape == (8,)
        assert terminated.shape == (8,)
        assert truncated.shape == (8,)
        assert "latent_Z" in step_info
        dr = env.do_reward(action)
        dt = env.do_transition(action)
        assert dr.shape[0] == 8
        assert dt.shape[0] == 8
