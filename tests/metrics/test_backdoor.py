"""Tests for the backdoor adjustment residual."""

from __future__ import annotations

import math

import torch

from causal_rl.metrics.backdoor import (
    backdoor_residual_per_arm,
    backdoor_residual_summary,
)


def test_residual_zero_when_z_unconfounding() -> None:
    """When Z is independent of A and only Z affects R, the naive estimator
    equals the back-door-adjusted estimator in expectation."""
    torch.manual_seed(0)
    n = 4000
    actions = torch.randint(0, 4, (n,))
    z = torch.randint(0, 2, (n,))
    # Reward depends only on Z, not on A — backdoor adjustment trivially
    # equals the naive estimator since A and Z are independent.
    rewards = z.float() + 0.05 * torch.randn(n)
    per_arm = backdoor_residual_per_arm(actions, rewards, z, n_actions=4)
    for v in per_arm.values():
        assert not math.isnan(v)
        assert abs(v) < 0.05, f"residual {v} should be near zero (no A-Z dep)"


def test_residual_recovers_confounding_bias() -> None:
    """When A and Z are correlated and Z affects R, the naive estimator is
    biased and the residual recovers the bias direction."""
    torch.manual_seed(0)
    n = 4000
    z = torch.randint(0, 2, (n,))
    # A is highly correlated with Z: action 0 prefers z=0, action 1 prefers z=1.
    actions = torch.where(
        torch.rand(n) < 0.85,
        z,  # match z 85% of the time
        1 - z,
    )
    rewards = z.float() + 0.05 * torch.randn(n)
    per_arm = backdoor_residual_per_arm(actions, rewards, z, n_actions=2)
    # Arm 0 has been confounded toward z=0 (low reward) → naive estimator
    # under-shoots the back-door average → residual is negative.
    # Arm 1 has been confounded toward z=1 (high reward) → naive overshoots
    # → residual is positive.
    assert per_arm[0] < -0.2, f"arm 0 residual {per_arm[0]} should be < -0.2"
    assert per_arm[1] > 0.2, f"arm 1 residual {per_arm[1]} should be > 0.2"


def test_residual_nan_when_z_hidden() -> None:
    actions = torch.randint(0, 4, (100,))
    rewards = torch.randn(100)
    per_arm = backdoor_residual_per_arm(actions, rewards, None, n_actions=4)
    for v in per_arm.values():
        assert math.isnan(v)


def test_summary_nan_when_z_hidden() -> None:
    actions = torch.randint(0, 4, (100,))
    rewards = torch.randn(100)
    s = backdoor_residual_summary(actions, rewards, None, n_actions=4)
    assert math.isnan(s["backdoor_residual_mean"])
    assert math.isnan(s["backdoor_residual_max"])
