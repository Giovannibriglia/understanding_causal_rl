from __future__ import annotations

from torch import Tensor

from causal_rl.algos.off_policy.dqn import DQN


class ConfoundedDQN(DQN):
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
        penalty = float(batch.get("delta_tv", 0.0))
        metrics["td_loss"] += self.sensitivity_lambda * penalty
        metrics["policy_loss"] += self.sensitivity_lambda * penalty
        return metrics
