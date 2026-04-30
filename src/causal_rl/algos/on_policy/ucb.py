from __future__ import annotations

import math

import torch
from torch import Tensor

from causal_rl.algos.base import BaseAlgorithm


class UCB(BaseAlgorithm):
    """Vanilla UCB1 (Auer 2002) for K-arm bandit problems."""

    kind = "on_policy"
    requires_continuous_action = False

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        self.n_actions = n_actions
        self._counts: Tensor = torch.zeros(n_actions, dtype=torch.float32)
        self._means: Tensor = torch.zeros(n_actions, dtype=torch.float32)
        self._t: int = 0

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        batch = obs.shape[0]
        device = obs.device

        if self._t == 0 or (self._counts == 0).any():
            # Pull each arm at least once, or purely random at t=0
            action = torch.randint(0, self.n_actions, (batch, 1), device=device)
        else:
            log_t = math.log(self._t)
            bonus = torch.sqrt(2.0 * log_t / self._counts.to(device))
            ucb = self._means.to(device) + bonus
            best = int(ucb.argmax().item())
            action = torch.full((batch, 1), best, dtype=torch.long, device=device)

        log_prob = torch.zeros(batch, device=device)
        return action, log_prob

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        actions = batch["action"].view(-1).to(torch.long)
        rewards = batch["reward"].view(-1).float()

        for a, r in zip(actions.tolist(), rewards.tolist(), strict=False):
            a_int = int(a)
            r_float = float(r)
            self._counts[a_int] += 1.0
            n = float(self._counts[a_int].item())
            self._means[a_int] += (r_float - float(self._means[a_int].item())) / n

        self._t += int(actions.shape[0])
        return {"td_loss": 0.0}
