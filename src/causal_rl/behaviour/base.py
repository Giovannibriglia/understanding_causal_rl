from __future__ import annotations

import abc
from typing import Literal

import torch
from torch import Tensor


class BiasedExplorer(abc.ABC):
    family: Literal["baseline", "reward_aligned", "information", "curiosity", "reward_misaligned"]
    requires_latent: bool
    # True when this policy conditions on U (the hidden confounder), creating a
    # spurious A↔R correlation that affects identifiability in pi_b_unknown cells.
    depends_on_u: bool
    bias_strength: float = 1.0

    def __init__(self, *args: object, bias_strength: float = 1.0, **kwargs: object) -> None:
        del args, kwargs
        self.bias_strength = float(bias_strength)

    def _mix_with_uniform(
        self,
        obs: Tensor,
        action: Tensor,
        log_prob: Tensor,
        n_actions: int | None = None,
        act_dim: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.bias_strength >= 1.0:
            return action, log_prob
        batch = obs.shape[0]
        device = obs.device
        if n_actions is not None:
            uniform_action = torch.randint(0, n_actions, (batch, 1), device=device)
            uniform_log_prob = torch.full((batch,), -torch.log(torch.tensor(float(n_actions))), device=device)
        else:
            assert act_dim is not None
            uniform_action = torch.rand((batch, act_dim), device=device)
            uniform_log_prob = torch.zeros((batch,), device=device)
        if self.bias_strength <= 0.0:
            return uniform_action, uniform_log_prob
        coin = (torch.rand((batch, 1), device=device) < self.bias_strength).to(action.dtype)
        mixed_action = coin * action + (1 - coin) * uniform_action
        mixed_log_prob = torch.where(
            coin.squeeze(-1) > 0,
            log_prob,
            uniform_log_prob,
        )
        return mixed_action, mixed_log_prob

    @abc.abstractmethod
    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
