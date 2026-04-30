from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class RewardAlignedExplorer(BiasedExplorer):
    family = "reward_aligned"

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        beta: float = 1.0,
        requires_latent: bool = False,
    ) -> None:
        self.n_actions = n_actions
        self.act_dim = act_dim
        self.beta = beta
        self.requires_latent = requires_latent
        if self.n_actions is not None:
            self._weights = torch.linspace(-0.5, 0.5, steps=self.n_actions)
        else:
            self._weights = torch.empty((0,), dtype=torch.float32)

    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.requires_latent and latent is None:
            msg = "Latent is required for confounded reward-aligned behaviour."
            raise ValueError(msg)
        if self.act_dim is not None:
            # Reward-aligned continuous behaviour:
            # push actions toward treatment intensity implied by the observed state.
            state_slice = obs[:, : self.act_dim]
            state_context = obs.mean(dim=-1, keepdim=True)
            score = 0.85 * state_slice + 0.15 * state_context
            base = torch.sigmoid(self.beta * score)
            if latent is not None:
                base = torch.sigmoid(torch.logit(base.clamp(1e-4, 1.0 - 1e-4)) + 0.10 * latent[:, :1])
            action = (base + 0.03 * torch.randn_like(base)).clamp(0.0, 1.0)
            log_prob = torch.zeros((obs.shape[0],), device=obs.device)
            return action, log_prob
        if self.n_actions is None:
            msg = "Either n_actions or act_dim must be provided."
            raise ValueError(msg)
        state_score = obs.mean(dim=-1, keepdim=True)
        weights = self._weights.to(obs.device).unsqueeze(0)
        scores = state_score * weights
        if latent is not None:
            scores = scores + latent[:, :1] * weights
        logits = self.beta * scores
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.unsqueeze(-1), dist.log_prob(action)
