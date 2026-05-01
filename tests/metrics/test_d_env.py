"""Tests for d_env_ks divergence metric (Phase E.2)."""

from __future__ import annotations

import torch

from causal_rl.envs.tabular_sepsis import TabularSepsisEnv
from causal_rl.metrics.d_env import d_env_ks


def _uniform_oracle(obs: torch.Tensor) -> torch.Tensor:
    """Always select action 0 (deterministic)."""
    return torch.zeros((obs.shape[0], 1), dtype=torch.long, device=obs.device)


def test_d_env_ks_same_env_near_zero() -> None:
    """d_env_ks(env, env, ...) should be ~0 since both envs are identical."""
    env = TabularSepsisEnv(cell=1, n_envs=16, device="cpu", alpha_conf=0.0)
    result = d_env_ks(env, env, _uniform_oracle, n_states=16, n_samples=50)
    assert result < 0.15, f"Same-env KS should be near 0, got {result:.4f}"


def test_d_env_ks_returns_float() -> None:
    """d_env_ks always returns a Python float."""
    env = TabularSepsisEnv(cell=1, n_envs=8, device="cpu", alpha_conf=0.0)
    result = d_env_ks(env, env, _uniform_oracle, n_states=8, n_samples=20)
    assert isinstance(result, float)
    assert result >= 0.0
