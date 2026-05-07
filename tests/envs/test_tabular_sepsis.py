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


def test_do_reward_marginalises_z_when_hidden() -> None:
    """v19: ``do_reward`` honours ``expose_z`` like ``sample_interventional``.

    With identical seeds across two envs whose only difference is ``expose_z``
    (C1 vs C3), the per-action reward distribution should differ on at least
    some entries — the C1 env conditions on the full latent state, the C3 env
    marginalises uniformly over Z.  Pre-v19 the two were bit-identical (both
    paths read ``reward_probs[self._latent_state, a]`` directly).
    """
    env_c1 = TabularSepsisEnv(cell=1, n_envs=64, device="cpu", alpha_conf=0.0)
    env_c3 = TabularSepsisEnv(cell=3, n_envs=64, device="cpu", alpha_conf=0.0)
    env_c1.reset(seed=0)
    env_c3.reset(seed=0)
    assert torch.equal(env_c1._latent_state, env_c3._latent_state), (
        "fixture broken: same seed should produce identical _latent_state"
    )
    action = torch.randint(0, 8, (64, 1))
    p_c1 = env_c1.do_reward(action)
    p_c3 = env_c3.do_reward(action)
    # The marginalisation does not move *every* entry (some latent states
    # have reward_probs[s_obs] ≈ reward_probs[OBS_STATES + s_obs] by
    # coincidence).  Asserting that the maximum elementwise gap exceeds
    # 1e-3 is enough to distinguish the marginalised and conditional paths.
    max_gap = (p_c1 - p_c3).abs().max().item()
    assert max_gap > 1e-3, (
        f"do_reward returned identical distributions across C1/C3 — "
        f"v19 marginalisation appears not to be active.  max gap: {max_gap:.6f}"
    )


def test_do_reward_matches_sample_interventional_in_expectation() -> None:
    """v19: ``do_reward`` and ``sample_interventional`` use the same Z-
    marginalisation when ``expose_z=False``.  Verify the two estimates of
    P(R=+1 | do(a)) agree to within a comfortable Monte-Carlo margin.

    With n=10000 the M.C. standard error on a Bernoulli is √(0.25/n) ≈ 0.005,
    so a 0.02 (≈4σ) tolerance is safe.
    """
    env = TabularSepsisEnv(cell=3, n_envs=8, device="cpu", alpha_conf=0.0)
    env.reset(seed=0)
    actions = torch.tensor([0, 1, 2, 3, 4]).view(-1, 1)
    for a_int in actions.tolist():
        a = torch.tensor(a_int * env.n_envs).view(env.n_envs, 1)
        analytical = env.do_reward(a)[:, 1]  # P(R=+1)
        samples = env.sample_interventional(state=None, action=a, n=10000)
        empirical = ((samples + 1.0) / 2.0).mean(dim=1)
        gap = (analytical - empirical).abs().max().item()
        assert gap < 0.02, (
            f"do_reward/sample_interventional disagree by {gap:.4f} on "
            f"action {a_int} — Z-marginalisation drift between the two paths"
        )
