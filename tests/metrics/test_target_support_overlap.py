"""Tests for target_support_overlap and pi_b_recovery_kl."""

from __future__ import annotations

import math

import torch

from causal_rl.metrics.coverage import pi_b_recovery_kl, target_support_overlap


def test_overlap_full_when_pi_b_uniform() -> None:
    """Uniform π_b assigns probability 1/n_actions > τ=1e-2 to every action,
    so any deterministic target lies entirely within π_b's support."""
    n, n_actions = 200, 4
    target_actions = torch.randint(0, n_actions, (n,))
    log_probs = torch.full((n, n_actions), -math.log(n_actions))
    overlap = target_support_overlap(target_actions, log_probs, tau=1e-2)
    assert float(overlap.item()) == 1.0


def test_overlap_zero_when_pi_b_avoids_target_action() -> None:
    """If π_b puts negligible mass on the action the target picks, overlap is 0."""
    n, n_actions = 200, 4
    target_actions = torch.zeros(n, dtype=torch.long)
    # π_b's log-prob on action 0 is -10 (≈ 4.5e-5), well below τ=1e-2.
    log_probs = torch.full((n, n_actions), 0.0)
    log_probs[:, 0] = -10.0
    overlap = target_support_overlap(target_actions, log_probs, tau=1e-2)
    assert float(overlap.item()) == 0.0


def test_kl_zero_when_predicted_equals_true() -> None:
    n, n_actions = 50, 4
    log_probs = torch.full((n, n_actions), -math.log(n_actions))
    kl = pi_b_recovery_kl(log_probs, log_probs)
    assert float(kl.item()) < 1e-6


def test_kl_nan_when_pi_b_unknown() -> None:
    n, n_actions = 50, 4
    log_probs = torch.full((n, n_actions), -math.log(n_actions))
    kl = pi_b_recovery_kl(log_probs, None)
    assert math.isnan(float(kl.item()))


def test_kl_positive_when_predicted_disagrees_with_true() -> None:
    n, n_actions = 50, 4
    log_uniform = torch.full((n, n_actions), -math.log(n_actions))
    # Concentrated prediction on action 0.
    log_pred = torch.full((n, n_actions), -10.0)
    log_pred[:, 0] = float(math.log(1.0 - 3 * math.exp(-10.0)))
    kl = pi_b_recovery_kl(log_pred, log_uniform)
    assert float(kl.item()) > 0.5
