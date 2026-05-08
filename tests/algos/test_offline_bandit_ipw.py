"""v21: ``OfflineBanditIPW`` — per-arm reward mean estimator with
inverse-propensity reweighting.

The IPW correction recovers ``E[r | do(A=a)]`` from confounded buffer
data when behaviour log-probs are available (true logprobs in
``pi_b_known=True`` cells, propensity-model estimates otherwise).
The fallback contract is that absent ``behaviour_logprob`` the algo
behaves like ``OfflineBanditNaive``.
"""

from __future__ import annotations

import math

import torch

from causal_rl.algos.off_policy.offline_bandit_ipw import OfflineBanditIPW
from causal_rl.algos.off_policy.offline_bandit_naive import OfflineBanditNaive


def test_uniform_logprobs_match_naive() -> None:
    """When all behaviour log-probs are uniform (``log(1/n_actions)``),
    every IPW weight equals ``n_actions``, so ``mu_hat = sum_w*r / sum_w``
    collapses to ``sum_r / count`` — identical to the naive estimator
    up to floating-point arithmetic."""
    torch.manual_seed(0)
    n_actions = 4
    n = 200
    actions = torch.randint(0, n_actions, (n,))
    rewards = torch.where(
        torch.rand(n) > 0.5, torch.ones(n), -torch.ones(n)
    )
    uniform_logp = torch.full((n,), math.log(1.0 / n_actions))

    ipw = OfflineBanditIPW(
        obs_dim=5, n_actions=n_actions, prior_count=0.0, prior_mean=0.0
    )
    naive = OfflineBanditNaive(
        obs_dim=5, n_actions=n_actions, prior_count=0.0, prior_mean=0.0
    )
    ipw.update({"action": actions, "reward": rewards, "behaviour_logprob": uniform_logp})
    naive.update({"action": actions, "reward": rewards})

    diff = (ipw.mu_hat - naive.mu_hat).abs().max().item()
    assert diff < 1e-5, (
        f"IPW with uniform logprobs should match naive within float "
        f"precision; got max gap {diff:.2e}"
    )


def test_ipw_recovers_true_means_under_confounded_sampling() -> None:
    """Confounded behaviour: arm 1 (true mean 0.5) sampled 800x, arms
    0/2/3 sampled 100/50/50x.  IPW with the true behaviour log-probs
    should recover ``E[r | do(A=a)]`` and pick arm 3 (true best).

    Reward is in {-1, +1} so true mean is in [-1, 1]; tolerance 0.20
    around true_means[3] = 0.9 (≈ 4σ on n=50, σ ≈ √(1/50) ≈ 0.14).
    """
    torch.manual_seed(0)
    n_actions = 4
    true_means = torch.tensor([0.1, 0.5, 0.3, 0.9])
    counts = [100, 800, 50, 50]
    n = sum(counts)
    actions = torch.cat(
        [torch.full((c,), a, dtype=torch.long) for a, c in enumerate(counts)]
    )
    p_pos = (1.0 + true_means[actions]) / 2.0
    rewards = torch.where(
        torch.bernoulli(p_pos) > 0,
        torch.ones(n),
        -torch.ones(n),
    )
    # True behaviour log-prob: log(count[a] / total).
    true_pi_b = torch.tensor([c / n for c in counts])
    true_logp = torch.log(true_pi_b[actions])

    ipw = OfflineBanditIPW(
        obs_dim=5, n_actions=n_actions, prior_count=0.0, prior_mean=0.0,
        ipw_clip=100.0,  # large enough not to clip on this fixture
    )
    ipw.update({"action": actions, "reward": rewards, "behaviour_logprob": true_logp})

    assert int(ipw.mu_hat.argmax().item()) == 3, (
        f"IPW with true logprobs should pick true best arm 3; "
        f"got argmax {int(ipw.mu_hat.argmax())}, mu_hat={ipw.mu_hat.tolist()}"
    )
    gap = abs(float(ipw.mu_hat[3].item()) - 0.9)
    assert gap < 0.20, (
        f"IPW estimate of arm 3 (true mean 0.9) should be within 0.20; "
        f"got mu_hat[3]={ipw.mu_hat[3].item():.3f} (gap {gap:.3f})"
    )


def test_ipw_clip_caps_extreme_weights() -> None:
    """A very small ``log π_b`` produces an explosive raw weight.  The
    clip at ``ipw_clip`` caps both the metric and the contribution to
    ``_count``."""
    n_actions = 4
    actions = torch.tensor([2, 0, 1])
    rewards = torch.tensor([1.0, -1.0, 1.0])
    # logp = -10 → raw weight ≈ 22026; logp = log(0.5) → weight 2; logp = log(0.25) → weight 4.
    logp = torch.tensor([-10.0, math.log(0.5), math.log(0.25)])

    ipw = OfflineBanditIPW(
        obs_dim=5, n_actions=n_actions, prior_count=0.0, prior_mean=0.0,
        ipw_clip=10.0,
    )
    metrics = ipw.update({"action": actions, "reward": rewards, "behaviour_logprob": logp})

    assert metrics["ipw_weight_max"] == 10.0, (
        f"raw weight ~22000 should be clipped to 10.0; got {metrics['ipw_weight_max']}"
    )
    # The arm-2 transition contributed weight 10 (clipped), not 22026.
    assert abs(float(ipw._count[2].item()) - 10.0) < 1e-6, (
        f"clip should cap _count[2] at 10.0 (the clipped weight), "
        f"got {ipw._count[2].item()}"
    )


def test_no_logprob_falls_back_to_naive() -> None:
    """A batch without ``behaviour_logprob`` should not raise — the IPW
    algo must remain well-defined when the runner mediates a cell
    where neither true nor estimated propensities are populated.

    The fallback contract: produce the same ``mu_hat`` as
    ``OfflineBanditNaive`` on the same batch.
    """
    torch.manual_seed(0)
    actions = torch.tensor([0, 1, 1, 2, 2, 2, 3])
    rewards = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])

    ipw = OfflineBanditIPW(
        obs_dim=5, n_actions=4, prior_count=0.0, prior_mean=0.0
    )
    naive = OfflineBanditNaive(
        obs_dim=5, n_actions=4, prior_count=0.0, prior_mean=0.0
    )
    # No ``behaviour_logprob`` key.
    ipw.update({"action": actions, "reward": rewards})
    naive.update({"action": actions, "reward": rewards})
    assert torch.allclose(ipw.mu_hat, naive.mu_hat), (
        f"fallback should match naive exactly; got "
        f"ipw={ipw.mu_hat.tolist()} naive={naive.mu_hat.tolist()}"
    )
