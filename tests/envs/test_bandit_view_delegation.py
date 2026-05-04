"""Regression tests for BanditView delegating sample_observational and
sample_interventional to the wrapped env.

Without this delegation, every bandit run dies at the first eval checkpoint
because ``compute_gap_metrics`` calls ``sample_observational`` on the bandit
view, which falls through to the base CausalEnv's NotImplementedError.
"""

from __future__ import annotations

import torch

from causal_rl.envs.bandit_view import BanditView
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv


def test_bandit_view_delegates_sample_observational() -> None:
    base = TabularSepsisEnv(cell=1, n_envs=4, device="cpu")
    bandit = BanditView(base)
    obs, _ = bandit.reset(seed=0)
    actions = torch.zeros(4, dtype=torch.long)
    samples = bandit.sample_observational(state=obs, action=actions, n=8)
    assert samples.shape == (4, 8)
    # Tabular sepsis returns rewards in {-1, +1}.
    unique = set(int(v) for v in samples.unique().tolist())
    assert unique <= {-1, 1}, f"unexpected reward values {unique}"


def test_bandit_view_delegates_sample_interventional() -> None:
    base = TabularSepsisEnv(cell=2, n_envs=4, device="cpu")
    bandit = BanditView(base)
    obs, _ = bandit.reset(seed=0)
    actions = torch.zeros(4, dtype=torch.long)
    samples = bandit.sample_interventional(state=obs, action=actions, n=4)
    assert samples.shape == (4, 4)


def test_bandit_view_gap_metrics_does_not_crash() -> None:
    """``compute_gap_metrics`` should run end-to-end on a BanditView.

    This is the actual failure mode that v8's smoke commentary flagged:
    the runner calls ``compute_gap_metrics`` at every eval checkpoint, and
    without sample_observational delegation it raises NotImplementedError.
    """
    from causal_rl.metrics.runner_hooks import compute_gap_metrics

    base = TabularSepsisEnv(cell=1, n_envs=4, device="cpu")
    bandit = BanditView(base)
    obs, _ = bandit.reset(seed=0)
    actions = torch.zeros((4, 1), dtype=torch.long)
    rewards = torch.zeros(4)
    metrics = compute_gap_metrics(
        bandit,
        obs,
        actions,
        observed_reward=rewards,
        pi_b_logprob=None,
        n_samples=8,
        n_bootstrap=0,
    )
    assert "delta_tv" in metrics
    assert metrics["delta_tv"] == metrics["delta_tv"]  # not NaN
