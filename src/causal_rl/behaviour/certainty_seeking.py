from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class CertaintySeekingExplorer(BiasedExplorer):
    family = "information"
    depends_on_u = False

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        gamma_info: float = 1.0,
        epsilon_mix: float = 0.05,
        requires_latent: bool = False,
    ) -> None:
        self.n_actions = n_actions
        self.act_dim = act_dim
        self.gamma_info = gamma_info
        self.epsilon_mix = epsilon_mix
        self.requires_latent = requires_latent
        if self.n_actions is not None:
            self._basis = torch.linspace(0.2, 1.2, steps=self.n_actions)
        else:
            self._basis = torch.empty((0,), dtype=torch.float32)

    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.requires_latent and latent is None:
            msg = "Latent is required for confounded certainty-seeking behaviour."
            raise ValueError(msg)
        if self.act_dim is not None:
            entropy_proxy = obs.std(dim=-1, keepdim=True)
            base = torch.sigmoid(-self.gamma_info * (obs[:, : self.act_dim] + 0.25 * entropy_proxy))
            if latent is not None:
                base = torch.sigmoid(base - 0.1 * latent[:, :1])
            action = (base - 0.2).clamp(0.0, 1.0)
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
        logits = -self.gamma_info * entropy_proxy
        probs = torch.softmax(logits, dim=-1)
        if self.epsilon_mix > 0.0:
            uniform = torch.full_like(probs, 1.0 / float(self.n_actions))
            probs = (1.0 - self.epsilon_mix) * probs + self.epsilon_mix * uniform
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action.unsqueeze(-1), dist.log_prob(action)
