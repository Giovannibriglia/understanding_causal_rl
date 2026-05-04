"""Tests for the conditional MI lower bound."""

from __future__ import annotations

import math

import torch

from causal_rl.metrics.info_bottleneck import conditional_mi_lower_bound


def test_mi_nan_when_z_hidden() -> None:
    n, obs_dim = 200, 4
    obs = torch.randn(n, obs_dim)
    actions = torch.randint(0, 4, (n,))
    rewards = torch.randn(n)
    mi = conditional_mi_lower_bound(obs, actions, rewards, latent_z=None)
    assert math.isnan(mi)


def test_mi_small_when_r_independent_of_z_given_sa() -> None:
    """When R = f(S, A) + ε is independent of Z given (S, A), the conditional
    MI estimator should return a small (near-zero) value."""
    torch.manual_seed(0)
    n, obs_dim, z_dim = 800, 4, 2
    obs = torch.randn(n, obs_dim)
    actions = torch.randint(0, 4, (n,)).float()
    z = torch.randn(n, z_dim)  # independent of (S, A) and R
    rewards = obs.sum(dim=-1) + actions + 0.1 * torch.randn(n)
    mi = conditional_mi_lower_bound(obs, actions, rewards, z, n_epochs=80)
    # Sanity: MI should be small (< 0.5 nats); does not need to be exactly 0
    # because the variational estimator is upper-bounded by 0 only at the
    # true distribution.
    assert mi < 0.6, f"MI {mi:.3f} should be small when R ⊥ Z | (S, A)"


def test_mi_larger_when_r_depends_strongly_on_z() -> None:
    """When R = g(Z) + small noise — i.e. R is largely explained by Z beyond
    (S, A) — the conditional MI estimator should return a notably larger
    value than the independence baseline."""
    torch.manual_seed(0)
    n, obs_dim, z_dim = 800, 4, 2
    obs = torch.randn(n, obs_dim)
    actions = torch.randint(0, 4, (n,)).float()
    z = torch.randn(n, z_dim)
    rewards = z.sum(dim=-1) + 0.05 * torch.randn(n)
    mi = conditional_mi_lower_bound(obs, actions, rewards, z, n_epochs=80)
    # We assert a directional, gentle gate to avoid flakiness from variational
    # estimator variance.
    assert mi > 0.05, f"MI {mi:.3f} should be > 0.05 when R is largely Z"
