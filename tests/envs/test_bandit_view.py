from __future__ import annotations

import torch

from causal_rl.envs.bandit_view import BanditView
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv


def _make_bandit(cell: int = 1, n_envs: int = 4) -> BanditView:
    env = TabularSepsisEnv(cell=cell, n_envs=n_envs)
    return BanditView(env)


def test_bandit_horizon_is_one() -> None:
    bandit = _make_bandit()
    assert bandit.horizon == 1


def test_bandit_step_returns_done() -> None:
    bandit = _make_bandit()
    bandit.reset()
    action = torch.zeros(bandit.n_envs, 1, dtype=torch.long)
    _obs, _reward, terminated, truncated, _info = bandit.step(action)
    assert terminated.all(), "BanditView.step should return terminated=True for all envs"
    assert not truncated.any(), "BanditView.step should return truncated=False for all envs"


def test_bandit_obs_shape_matches_env() -> None:
    inner = TabularSepsisEnv(cell=1, n_envs=4)
    bandit = BanditView(inner)
    assert bandit.obs_shape == inner.obs_shape

    obs, _ = bandit.reset()
    assert obs.shape[-len(inner.obs_shape) :] == inner.obs_shape
