"""Tests for d_env_ks divergence metric (Phase E.2)."""

from __future__ import annotations

import torch

from causal_rl.envs.perturbations import (
    TABULAR_DIAGONAL,
    perturb_tabular_rewards,
    perturb_tabular_transition,
)
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv
from causal_rl.metrics.d_env import d_env_ks


def _uniform_oracle(obs: torch.Tensor) -> torch.Tensor:
    """Always select action 0 (deterministic)."""
    return torch.zeros((obs.shape[0], 1), dtype=torch.long, device=obs.device)


def _make_perturbed(
    cell: int, n_envs: int, eps_t: float, eps_r: float
) -> TabularSepsisEnv:  # noqa: N803
    env = TabularSepsisEnv(cell=cell, n_envs=n_envs, device="cpu", alpha_conf=0.0)
    env.transition = perturb_tabular_transition(env.transition, eps_t)
    env.reward_probs = perturb_tabular_rewards(env.reward_probs, eps_r)
    return env


def test_d_env_ks_same_env_near_zero() -> None:
    """d_env_ks(env, env, ...) should be ≈0 since both envs are identical."""
    env = TabularSepsisEnv(cell=1, n_envs=16, device="cpu", alpha_conf=0.0)
    result = d_env_ks(env, env, _uniform_oracle, n_states=16, n_samples=100)
    assert result < 0.10, f"Same-env KS should be near 0, got {result:.4f}"


def test_d_env_ks_returns_float() -> None:
    """d_env_ks always returns a Python float."""
    env = TabularSepsisEnv(cell=1, n_envs=8, device="cpu", alpha_conf=0.0)
    result = d_env_ks(env, env, _uniform_oracle, n_states=8, n_samples=20)
    assert isinstance(result, float)
    assert result >= 0.0


def test_d_env_ks_max_perturbation_nonzero() -> None:
    """d_env_ks should be > 0.1 at the largest perturbation level."""
    spec = TABULAR_DIAGONAL[-1]  # eps_T=0.5, eps_R=0.3
    env_train = TabularSepsisEnv(cell=1, n_envs=32, device="cpu", alpha_conf=0.0)
    env_target = _make_perturbed(1, 32, spec.eps_T, spec.eps_R)  # type: ignore[call-arg]
    result = d_env_ks(env_train, env_target, _uniform_oracle, n_states=32, n_samples=200)
    assert result > 0.1, f"Max-perturbation KS should be > 0.1, got {result:.4f}"


def test_d_env_ks_monotone_in_perturbation() -> None:
    """d_env_ks should be non-decreasing along the TABULAR_DIAGONAL schedule."""
    n_envs = 16
    env_train = TabularSepsisEnv(cell=1, n_envs=n_envs, device="cpu", alpha_conf=0.0)
    prev = 0.0
    for spec in TABULAR_DIAGONAL:
        env_target = _make_perturbed(1, n_envs, spec.eps_T, spec.eps_R)
        val = d_env_ks(env_train, env_target, _uniform_oracle, n_states=n_envs, n_samples=100)
        assert (
            val >= prev - 0.02
        ), f"KS={val:.3f} < prev={prev:.3f} at eps_T={spec.eps_T}; should be non-decreasing"
        prev = val
