from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.algos.base import BaseAlgorithm


class RCT(BaseAlgorithm):
    """Uniformly random arm selection — equivalent to a randomised control trial."""

    kind = "on_policy"
    requires_continuous_action = False

    def __init__(self, obs_dim: int, n_actions: int) -> None:
        self.n_actions = n_actions

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        batch = obs.shape[0]
        device = obs.device
        action = torch.randint(0, self.n_actions, (batch, 1), device=device)
        log_prob = torch.zeros(batch, device=device)
        return action, log_prob

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        return {}
