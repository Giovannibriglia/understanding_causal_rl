from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class UniformExplorer(BiasedExplorer):
    family = "baseline"
    requires_latent = False
    depends_on_u = False

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        requires_latent: bool | None = None,
    ) -> None:
        del requires_latent
        self.n_actions = n_actions
        self.act_dim = act_dim

    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch = obs.shape[0]
        device = obs.device
        if self.n_actions is not None:
            probs = torch.full((batch, self.n_actions), 1.0 / self.n_actions, device=device)
            dist = Categorical(probs=probs)
            action = dist.sample().unsqueeze(-1)
            log_prob = dist.log_prob(action.squeeze(-1))
            return action, log_prob
        if self.act_dim is None:
            msg = "Either n_actions or act_dim must be provided."
            raise ValueError(msg)
        action = torch.rand((batch, self.act_dim), device=device)
        log_prob = torch.zeros((batch,), device=device)
        return action, log_prob
