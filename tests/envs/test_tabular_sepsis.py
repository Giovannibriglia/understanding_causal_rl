from __future__ import annotations

import torch

from causal_rl.envs.tabular_sepsis import TabularSepsisEnv
from causal_rl.metrics.plugin import plugin_tv_gap


def test_tabular_sepsis_cell3_plugin_gap_near_zero() -> None:
    env = TabularSepsisEnv(cell=3, n_envs=64, device="cpu")
    _obs, _ = env.reset(seed=0)
    action = torch.randint(0, 8, (64, 1))
    gap = plugin_tv_gap(env, action, pi_b_logprob=None)
    assert float(gap.mean().item()) < 1e-6


def test_tabular_sepsis_cell7_gap_bounded() -> None:
    env = TabularSepsisEnv(cell=7, n_envs=128, alpha_conf=2.0, device="cpu")
    _obs, _ = env.reset(seed=0)
    action = torch.randint(0, 8, (128, 1))
    gap = plugin_tv_gap(env, action, pi_b_logprob=None)
    lam = torch.exp(torch.tensor(2.0))
    bound = (lam - 1.0) / (lam + 1.0)
    assert float(gap.mean().item()) <= float(bound.item()) + 1e-2
