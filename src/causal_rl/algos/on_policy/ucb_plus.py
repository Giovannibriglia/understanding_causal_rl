from __future__ import annotations

import math

import torch
from torch import Tensor

from causal_rl.algos.on_policy.ucb import UCB


class UCBPlus(UCB):
    """Bareinboim Algorithm 23 style UCB with natural-bound clipping.

    The UCB index for arm *a* is clipped into ``[lower_bounds[a], upper_bounds[a]]``
    before arm selection.  This allows prior causal knowledge about the range of
    achievable rewards to shrink the effective exploration bonus.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
    ) -> None:
        super().__init__(obs_dim, n_actions)
        if lower_bounds is not None:
            self._lower: Tensor | None = torch.tensor(lower_bounds, dtype=torch.float32)
        else:
            self._lower = torch.zeros(n_actions, dtype=torch.float32)

        if upper_bounds is not None:
            self._upper: Tensor | None = torch.tensor(upper_bounds, dtype=torch.float32)
        else:
            self._upper = torch.ones(n_actions, dtype=torch.float32)

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        batch = obs.shape[0]
        device = obs.device

        if self._t == 0 or (self._counts == 0).any():
            action = torch.randint(0, self.n_actions, (batch, 1), device=device)
        else:
            log_t = math.log(self._t)
            bonus = torch.sqrt(2.0 * log_t / self._counts.to(device))
            ucb = self._means.to(device) + bonus

            # Clip each arm's index into its natural bounds
            lower = (
                self._lower.to(device)
                if self._lower is not None
                else torch.zeros(self.n_actions, device=device)
            )
            upper = (
                self._upper.to(device)
                if self._upper is not None
                else torch.ones(self.n_actions, device=device)
            )
            ucb = torch.max(torch.min(ucb, upper), lower)

            best = int(ucb.argmax().item())
            action = torch.full((batch, 1), best, dtype=torch.long, device=device)

        log_prob = torch.zeros(batch, device=device)
        return action, log_prob

    def n_informative_bounds(self) -> int:
        # An arm is informative when its upper bound is strictly below the
        # best lower bound across arms — i.e., the arm is provably suboptimal
        # under the natural bounds (Bareinboim Thm 9.2.2).
        if self._lower is None or self._upper is None:
            return 0
        mu_lower_max = float(self._lower.max().item())
        return int((self._upper < mu_lower_max - 1e-9).sum().item())
