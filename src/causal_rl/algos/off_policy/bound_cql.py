from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.algos.off_policy.cql import CQL


class BoundCQL(CQL):
    """CQL variant that clips bootstrapped Q-targets into natural-bounds intervals.

    For each action *a* the TD target is clipped into
    ``[lower_bounds[a] / (1 - gamma), upper_bounds[a] / (1 - gamma)]``,
    reflecting known causal bounds on the achievable cumulative reward.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 1.0,
        tau: float = 0.01,
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
    ) -> None:
        super().__init__(obs_dim, n_actions, lr=lr, gamma=gamma, alpha=alpha, tau=tau)
        self._lower: Tensor | None = (
            torch.tensor(lower_bounds, dtype=torch.float32) if lower_bounds is not None else None
        )
        self._upper: Tensor | None = (
            torch.tensor(upper_bounds, dtype=torch.float32) if upper_bounds is not None else None
        )

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        action = batch["action"].view(-1).to(torch.long)
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        q_all = self.q(obs)
        q_sa = q_all.gather(1, action.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q = self.target_q(next_obs).max(dim=-1).values
            target = reward + self.gamma * (1.0 - done) * next_q

            # Clip TD targets per action using natural bounds
            if self._lower is not None and self._upper is not None:
                lower = self._lower.to(obs.device)
                upper = self._upper.to(obs.device)
                lo = lower[action] / (1.0 - self.gamma)
                hi = upper[action] / (1.0 - self.gamma)
                target = torch.max(torch.min(target, hi), lo)

        td_loss = ((q_sa - target) ** 2).mean()
        conservative = (torch.logsumexp(q_all, dim=-1) - q_sa).mean()
        loss = td_loss + self.alpha * conservative

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0).item()
        self.optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target_q.parameters(), strict=True):
                tp.mul_(1.0 - self.tau)
                tp.add_(self.tau * p)

        return {
            "td_loss": float(td_loss.item()),
            "policy_loss": float((self.alpha * conservative).item()),
            "value_loss": float(td_loss.item()),
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": float(grad_norm),
        }

    def n_informative_bounds(self) -> int:
        if self._lower is None or self._upper is None:
            return 0
        span = self._upper - self._lower
        return int((span < 1.0 - 1e-6).sum().item())
