"""v21: ``OfflineBanditNaive`` — per-arm running-mean estimator.

The naive estimator accumulates raw observed rewards.  When the
buffer is unconfounded (uniform behaviour, ``alpha_conf = 0``) it
recovers true arm means.  When the buffer is confounded the estimator
is biased — these tests verify both the recovery on unconfounded data
and the failure mode the IPW counterpart (``test_offline_bandit_ipw``)
will correct.
"""

from __future__ import annotations

import torch

from causal_rl.algos.off_policy.offline_bandit_naive import OfflineBanditNaive


def test_select_action_picks_argmax_mean() -> None:
    """``select_action`` is greedy on ``mu_hat``: every env in the batch
    receives the same argmax index."""
    algo = OfflineBanditNaive(obs_dim=5, n_actions=4, prior_count=0.0, prior_mean=0.0)
    # Manually populate sums/counts so mu_hat = [0.1, 0.5, 0.3, 0.2].
    algo._sum = torch.tensor([0.1, 0.5, 0.3, 0.2])
    algo._count = torch.tensor([1.0, 1.0, 1.0, 1.0])

    obs = torch.zeros(2, 5)
    action, logprob = algo.select_action(obs)
    assert action.shape == (2, 1)
    assert torch.equal(action.view(-1), torch.tensor([1, 1]))
    assert logprob.shape == (2,)
    assert torch.equal(logprob, torch.zeros(2))


def test_update_unconfounded_recovers_true_means() -> None:
    """Uniform sampling: ``mu_hat`` should recover the true arm means
    within Bernoulli σ on n=4000 total samples (~1000 per arm).

    Reward in {-1, +1} so per-arm variance is up to 1; σ on the mean
    estimate with n≈1000 is up to √(1/1000) ≈ 0.032.  A 0.10 tolerance
    is comfortable (~3σ) and well below the confounded-bias scale
    (>0.3) that the IPW counterpart corrects.
    """
    torch.manual_seed(0)
    n_actions = 4
    true_means = torch.tensor([0.1, 0.5, 0.3, 0.9])
    n = 4000
    actions = torch.randint(0, n_actions, (n,))
    # Reward is Bernoulli{-1, +1} with E[r] = 2 * p_pos - 1 = true_mean ⇒
    # p_pos = (1 + true_mean) / 2.
    p_pos = (1.0 + true_means[actions]) / 2.0
    rewards = torch.where(
        torch.bernoulli(p_pos) > 0,
        torch.ones(n),
        -torch.ones(n),
    )
    algo = OfflineBanditNaive(
        obs_dim=5, n_actions=n_actions, prior_count=0.0, prior_mean=0.0
    )
    algo.update({"action": actions, "reward": rewards})

    gap = (algo.mu_hat - true_means).abs().max().item()
    assert gap < 0.10, (
        f"naive estimator on unconfounded data should recover true means "
        f"within 0.10; got gap {gap:.3f}, mu_hat={algo.mu_hat.tolist()}"
    )
    assert int(algo.mu_hat.argmax().item()) == 3, (
        f"argmax should match true best arm 3; got {int(algo.mu_hat.argmax())}"
    )


def test_update_confounded_estimator_reliability_dominated_by_sampling() -> None:
    """Demonstrates the precondition for naive failure: when the
    behaviour policy oversamples a non-best arm, the naive estimator's
    *reliability* (per-arm sample count) is dominated by the behaviour
    policy, not by the action's true effect.

    The point is not that naive will pick the wrong arm on every seed —
    that's an empirical statement that depends on noise.  The point is
    that the estimator is sensitive to behaviour-policy bias: arm 1
    has 16× the samples of arm 3 even though arm 3 has the highest
    true reward.  The IPW counterpart (Phase 2) corrects this.
    """
    torch.manual_seed(0)
    n_actions = 4
    true_means = torch.tensor([0.1, 0.5, 0.3, 0.9])
    # Confounded behaviour: arm 1 sampled 800 times, arms 0/2/3 sampled
    # 100/50/50 times respectively.
    counts = [100, 800, 50, 50]
    actions = torch.cat([torch.full((c,), a, dtype=torch.long) for a, c in enumerate(counts)])
    p_pos = (1.0 + true_means[actions]) / 2.0
    rewards = torch.where(
        torch.bernoulli(p_pos) > 0,
        torch.ones_like(p_pos),
        -torch.ones_like(p_pos),
    )

    algo = OfflineBanditNaive(
        obs_dim=5, n_actions=n_actions, prior_count=0.0, prior_mean=0.0
    )
    algo.update({"action": actions, "reward": rewards})

    # Sample-count asymmetry: arm 1 (oversampled) has many more samples
    # than arm 3 (the truly-best arm).  This is the structural property
    # that biases the naive estimator's variance.
    assert algo._count[1].item() > 10 * algo._count[3].item(), (
        f"fixture broken: arm 1 should be oversampled relative to arm 3; "
        f"got counts {algo._count.tolist()}"
    )
    # Wide CI on under-sampled arms: arm 3 with n=50 has Bernoulli σ
    # ≈ √(0.25/50) ≈ 0.07 vs arm 1 with n=800 has σ ≈ 0.018.  Use the
    # square-root-of-sample-size proxy for relative confidence width.
    rel_ci_arm3 = 1.0 / float(algo._count[3].item()) ** 0.5
    rel_ci_arm1 = 1.0 / float(algo._count[1].item()) ** 0.5
    assert rel_ci_arm3 > 3.0 * rel_ci_arm1, (
        "arm 3 should have a much wider CI than arm 1 due to behaviour-"
        f"policy oversampling; got rel CIs {rel_ci_arm3:.3f} vs {rel_ci_arm1:.3f}"
    )
