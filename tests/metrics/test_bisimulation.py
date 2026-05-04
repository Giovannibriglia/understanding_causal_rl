"""Tests for the bisimulation distance."""

from __future__ import annotations

import torch

from causal_rl.metrics.bisimulation import (
    bisimulation_distance,
    bisimulation_distance_diag,
)


def _make_random_mdp(n_states: int, n_actions: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    rewards = torch.rand((n_states, n_actions), generator=g)
    transitions = torch.rand((n_states, n_actions, n_states), generator=g)
    transitions = transitions / transitions.sum(dim=-1, keepdim=True)
    return rewards, transitions


def test_distance_zero_when_envs_identical() -> None:
    rew, t = _make_random_mdp(6, 3, seed=0)
    d = bisimulation_distance(rew, t, rew, t, gamma=0.5, n_iters=20)
    assert torch.diagonal(d).abs().max().item() < 1e-3


def test_distance_grows_with_reward_shift() -> None:
    rew, t = _make_random_mdp(6, 3, seed=1)
    d_small = bisimulation_distance(rew, t, rew + 0.05, t, gamma=0.5, n_iters=10)
    d_large = bisimulation_distance(rew, t, rew + 0.5, t, gamma=0.5, n_iters=10)
    assert torch.diagonal(d_large).mean().item() > torch.diagonal(d_small).mean().item()


def test_diag_is_finite_and_non_negative() -> None:
    rew, t = _make_random_mdp(8, 2, seed=2)
    val = bisimulation_distance_diag(rew, t, rew, t, gamma=0.7, n_iters=8)
    assert val >= 0.0
    assert val == val  # not NaN
