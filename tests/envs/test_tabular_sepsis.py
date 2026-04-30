from __future__ import annotations

import torch

from causal_rl.envs.tabular_sepsis import TabularSepsisEnv
from causal_rl.metrics.runner_hooks import compute_gap_metrics


def test_tabular_sepsis_cell1_gap_converges_to_zero() -> None:
    """Cell 1: expose_Z, uniform behaviour policy, alpha_conf=0.

    Δ̂_TV should converge toward zero as n_samples grows, since there is no
    selection bias (alpha_conf=0) and the observational distribution equals the
    interventional distribution.  We verify that the gap with large n is
    smaller than with small n.
    """
    env_small = TabularSepsisEnv(cell=1, n_envs=32, device="cpu", alpha_conf=0.0)
    env_large = TabularSepsisEnv(cell=1, n_envs=32, device="cpu", alpha_conf=0.0)
    env_small.reset(seed=0)
    env_large.reset(seed=0)
    action = torch.randint(0, 8, (32, 1))

    metrics_small = compute_gap_metrics(
        env_small,
        None,
        action,
        observed_reward=None,
        pi_b_logprob=None,
        n_samples=50,
        n_bootstrap=0,
    )
    metrics_large = compute_gap_metrics(
        env_large,
        None,
        action,
        observed_reward=None,
        pi_b_logprob=None,
        n_samples=2000,
        n_bootstrap=0,
    )
    # With no confounding, the gap should be small and should shrink with n.
    assert metrics_large["delta_tv"] <= metrics_small["delta_tv"] + 0.05, (
        f"Gap did not shrink with larger n: small={metrics_small['delta_tv']:.4f}, "
        f"large={metrics_large['delta_tv']:.4f}"
    )
    # With large n and no confounding, the gap should be near zero.
    assert (
        metrics_large["delta_tv"] < 0.1
    ), f"Gap too large for unconfounded cell 1: {metrics_large['delta_tv']:.4f}"


def test_tabular_sepsis_cell4_gap_positive_with_confounding() -> None:
    """Cell 4 (pomdp_unknown): alpha_conf=2.0.

    Δ̂_TV should stabilise at a positive value as n_samples grows, reflecting
    the selection bias introduced by the confounding behaviour policy.
    """
    env = TabularSepsisEnv(cell=4, n_envs=128, alpha_conf=2.0, device="cpu")
    env.reset(seed=0)
    action = torch.randint(0, 8, (128, 1))

    metrics = compute_gap_metrics(
        env, None, action, observed_reward=None, pi_b_logprob=None, n_samples=1000, n_bootstrap=0
    )
    # With alpha_conf=2.0 the gap should be measurably positive.
    assert (
        metrics["delta_tv"] > 0.01
    ), f"Gap expected positive for confounded cell 7, got {metrics['delta_tv']:.4f}"
    # And bounded by the Λ-sensitivity bound: (λ-1)/(λ+1) where λ=exp(alpha_conf).
    lam = torch.exp(torch.tensor(2.0))
    bound = float(((lam - 1.0) / (lam + 1.0)).item())
    assert (
        metrics["delta_tv"] <= bound + 0.05
    ), f"Gap {metrics['delta_tv']:.4f} exceeds Λ-bound {bound:.4f}"


def test_tabular_sepsis_cell4_gap_bounded() -> None:
    """Legacy bound check: gap for cell 4 does not exceed the Λ-sensitivity bound."""
    env = TabularSepsisEnv(cell=4, n_envs=128, alpha_conf=2.0, device="cpu")
    env.reset(seed=0)
    action = torch.randint(0, 8, (128, 1))
    metrics = compute_gap_metrics(
        env, None, action, observed_reward=None, pi_b_logprob=None, n_samples=500, n_bootstrap=0
    )
    lam = torch.exp(torch.tensor(2.0))
    bound = float(((lam - 1.0) / (lam + 1.0)).item())
    assert metrics["delta_tv"] <= bound + 0.05


def test_tabular_sepsis_ci_returned() -> None:
    """Bootstrap CI should be returned and be a valid interval."""
    env = TabularSepsisEnv(cell=4, n_envs=16, alpha_conf=2.0, device="cpu")
    env.reset(seed=42)
    action = torch.randint(0, 8, (16, 1))
    metrics = compute_gap_metrics(
        env, None, action, observed_reward=None, pi_b_logprob=None, n_samples=100, n_bootstrap=50
    )
    assert "delta_tv_ci_lo" in metrics
    assert "delta_tv_ci_hi" in metrics
    assert metrics["delta_tv_ci_lo"] <= metrics["delta_tv"] + 1e-6
    assert metrics["delta_tv_ci_hi"] >= metrics["delta_tv"] - 1e-6
