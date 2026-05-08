from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.algos.base import BaseAlgorithm


class OfflineBanditNaive(BaseAlgorithm):
    """Per-arm reward mean estimator, no IPW correction.

    Maintains running estimates ``mu_hat[a] = sum_r / count`` for each
    arm based on raw observed rewards in the offline buffer.  At eval
    time picks ``argmax_a mu_hat[a]``.

    On confounded data (behaviour policy depends on latent U with
    ``alpha_conf > 0``) this estimator is biased: arms preferentially
    sampled by the behaviour policy have ``mu_hat`` that reflects the
    conditional reward given the behaviour's selection, not the
    interventional reward.  Used as the ``pi_b_known``-axis baseline
    for the bandit IPW counterpart.

    Non-contextual: the algorithm ignores ``obs`` deliberately.  The
    cell-config's ``expose_z`` axis is exercised entirely through the
    env's reward distribution, which the buffer reflects.
    """

    kind = "off_policy"
    requires_continuous_action = False

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        prior_count: float = 1.0,
        prior_mean: float = 0.0,
    ) -> None:
        del obs_dim  # bandit estimator is non-contextual
        self.n_actions = n_actions
        # Beta-style priors so the first ``select_action`` is
        # well-defined before any update fires.
        self._sum = torch.full((n_actions,), float(prior_count) * float(prior_mean))
        self._count = torch.full((n_actions,), float(prior_count))

    @property
    def mu_hat(self) -> Tensor:
        return self._sum / self._count.clamp(min=1e-9)

    def select_action(
        self, obs: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        del deterministic  # always greedy
        batch = obs.shape[0]
        mu = self.mu_hat.to(obs.device)
        action = torch.argmax(mu).repeat(batch).unsqueeze(-1)
        # log P(action_taken) = 0 for the deterministic argmax policy.
        logprob = torch.zeros(batch, device=obs.device)
        return action, logprob

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        action = batch["action"].view(-1).to(torch.long).cpu()
        reward = batch["reward"].view(-1).cpu()
        self._sum = self._sum.scatter_add(
            0, action, reward.to(self._sum.dtype)
        )
        ones = torch.ones_like(reward, dtype=self._count.dtype)
        self._count = self._count.scatter_add(0, action, ones)
        with torch.no_grad():
            mu_var = float(self.mu_hat.var().item())
        return {
            "td_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": mu_var,
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": 0.0,
        }
