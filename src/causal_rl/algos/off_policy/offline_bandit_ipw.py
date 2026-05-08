from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.algos.off_policy.offline_bandit_naive import OfflineBanditNaive


class OfflineBanditIPW(OfflineBanditNaive):
    """Per-arm reward mean estimator with IPW correction.

    For each transition ``(a, r, log π_b(a))`` in the buffer, accumulates
    ``w * r`` into ``_sum[a]`` and ``w`` into ``_count[a]``, where
    ``w = exp(-log π_b(a))`` clamped to ``ipw_clip``.  The resulting
    ``mu_hat[a] = sum / count`` is an IPW estimator of
    ``E[r | do(A=a)]`` — unbiased when ``log π_b`` is the true
    behaviour log-prob, consistent when it is a well-calibrated
    estimate.

    The runner mediates per-cell data quality: when
    ``pi_b_known=True`` the batch's ``behaviour_logprob`` is the true
    behaviour log-prob; when ``pi_b_known=False`` it's the propensity
    model's estimate.  The algo uses whichever is present, falling
    back to naive accumulation when the field is absent.

    Pre-clamp the importance weight at ``ipw_clip`` to control
    variance: a behaviour-policy probability of 0.001 produces a
    weight of 1000, which dominates the estimator.  Clipping at 10
    matches the runner's ``compute_gap_metrics`` IPW path.
    """

    kind = "off_policy"
    requires_continuous_action = False

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        prior_count: float = 1.0,
        prior_mean: float = 0.0,
        ipw_clip: float = 10.0,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            n_actions=n_actions,
            prior_count=prior_count,
            prior_mean=prior_mean,
        )
        self.ipw_clip = ipw_clip

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        logp = batch.get("behaviour_logprob", None)
        if logp is None:
            # No behaviour log-prob — fall back to naive.  v22 configs
            # should set ``pi_b_known`` consistently with the cell, so
            # this branch is reached only on misconfigured runs.
            return super().update(batch)
        action = batch["action"].view(-1).to(torch.long).cpu()
        reward = batch["reward"].view(-1).cpu()
        weights = torch.exp(-logp.view(-1).cpu()).clamp(max=self.ipw_clip)
        weighted_reward = weights * reward.to(weights.dtype)
        self._sum = self._sum.scatter_add(
            0, action, weighted_reward.to(self._sum.dtype)
        )
        self._count = self._count.scatter_add(
            0, action, weights.to(self._count.dtype)
        )
        with torch.no_grad():
            mu_var = float(self.mu_hat.var().item())
            w_max = float(weights.max().item())
            w_mean = float(weights.mean().item())
        return {
            "td_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": mu_var,
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": 0.0,
            "ipw_weight_max": w_max,
            "ipw_weight_mean": w_mean,
        }
