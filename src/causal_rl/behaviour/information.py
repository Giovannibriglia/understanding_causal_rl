from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class InformationExplorer(BiasedExplorer):
    family = "information"

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        gamma_info: float = 1.0,
        requires_latent: bool = False,
    ) -> None:
        self.n_actions = n_actions
        self.act_dim = act_dim
        self.gamma_info = gamma_info
        self.requires_latent = requires_latent
        if self.n_actions is not None:
            self._basis = torch.linspace(0.2, 1.2, steps=self.n_actions)
        else:
            self._basis = torch.empty((0,), dtype=torch.float32)

    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.requires_latent and latent is None:
            msg = "Latent is required for confounded information behaviour."
            raise ValueError(msg)
        if self.act_dim is not None:
            # Information-seeking continuous behaviour:
            # upweight actions in uncertain/ambiguous regimes (near 0.5) and variable states.
            state_slice = obs[:, : self.act_dim]
            ambiguity = (1.0 - 2.0 * (state_slice - 0.5).abs()).clamp(0.0, 1.0)
            entropy_proxy = obs.std(dim=-1, keepdim=True)
            score = ambiguity + 0.30 * entropy_proxy - 0.50
            base = torch.sigmoid(self.gamma_info * score)
            if latent is not None:
                base = torch.sigmoid(torch.logit(base.clamp(1e-4, 1.0 - 1e-4)) + 0.10 * latent[:, :1])
            action = (base + 0.03 * torch.randn_like(base)).clamp(0.0, 1.0)
            log_prob = torch.zeros((obs.shape[0],), device=obs.device)
            return action, log_prob
        if self.n_actions is None:
            msg = "Either n_actions or act_dim must be provided."
            raise ValueError(msg)
        b = obs.shape[0]
        basis = self._basis.to(obs.device).unsqueeze(0).expand(b, -1)
        state_mag = obs.std(dim=-1, keepdim=True)
        entropy_proxy = state_mag * basis
        if latent is not None:
            entropy_proxy = entropy_proxy + 0.25 * latent[:, :1] * basis
        logits = self.gamma_info * entropy_proxy
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.unsqueeze(-1), dist.log_prob(action)
