from __future__ import annotations

from torch import Tensor

from causal_rl.algos.off_policy.dqn import DQN


class ConfoundedDQN(DQN):
    """DQN with a causal-sensitivity penalty on the TD loss.

    The penalty is the within-batch estimate of Δ̂_φ (the identifiability gap)
    passed in via ``batch["delta_tv"]``.  This is populated by the runner with
    the most recently evaluated gap metric from the eval env, so the penalty
    reflects a real divergence between the observational and interventional
    reward distributions rather than a proxy such as latent magnitude.

    Concretely the loss becomes:
        L = L_TD + sensitivity_lambda * Δ̂_φ
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        epsilon: float = 0.1,
        double: bool = True,
        sensitivity_lambda: float = 0.5,
    ) -> None:
        super().__init__(
            obs_dim=obs_dim,
            n_actions=n_actions,
            lr=lr,
            gamma=gamma,
            tau=tau,
            epsilon=epsilon,
            double=double,
        )
        self.sensitivity_lambda = sensitivity_lambda

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        metrics = super().update(batch)
        # delta_tv is the Δ̂_φ estimate (TV between observational and interventional
        # reward distributions) supplied by the runner from the eval env.
        # It is a scalar float; super().update() already computes the base loss.
        delta_phi = float(batch.get("delta_tv", 0.0))
        penalty = self.sensitivity_lambda * delta_phi
        metrics["td_loss"] = metrics.get("td_loss", 0.0) + penalty
        metrics["policy_loss"] = metrics.get("policy_loss", 0.0) + penalty
        metrics["causal_penalty"] = penalty
        return metrics
