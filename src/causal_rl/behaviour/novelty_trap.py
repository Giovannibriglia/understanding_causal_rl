from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class NoveltyTrapExplorer(BiasedExplorer):
    family = "curiosity"

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        delta: float = 1.0,
        risk_strength: float = 1.0,
        epsilon_mix: float = 0.05,
        requires_latent: bool = False,
    ) -> None:
        self.n_actions = n_actions
        self.act_dim = act_dim
        self.delta = delta
        self.risk_strength = risk_strength
        self.epsilon_mix = epsilon_mix
        self.requires_latent = requires_latent
        if self.n_actions is not None:
            self._counts = torch.ones((self.n_actions,), dtype=torch.float32)
            self._risky_bias = torch.linspace(0.0, 1.0, steps=self.n_actions)
        else:
            self._counts = torch.empty((0,), dtype=torch.float32)
            self._risky_bias = torch.empty((0,), dtype=torch.float32)

    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.requires_latent and latent is None:
            msg = "Latent is required for confounded novelty-trap behaviour."
            raise ValueError(msg)
        if self.act_dim is not None:
            state_dev = (obs[:, : self.act_dim] - 0.5).abs()
            base = torch.sigmoid(self.delta * state_dev + self.risk_strength * 0.8)
            if latent is not None:
                base = torch.sigmoid(base + 0.1 * latent[:, :1])
            action = (base + 0.05 * torch.randn_like(base)).clamp(0.0, 1.0)
            log_prob = torch.zeros((obs.shape[0],), device=obs.device)
            return action, log_prob
        if self.n_actions is None:
            msg = "Either n_actions or act_dim must be provided."
            raise ValueError(msg)
        counts = self._counts.to(obs.device)
        novelty = 1.0 / torch.sqrt(counts)
        risky = self._risky_bias.to(obs.device)
        logits = self.delta * novelty.unsqueeze(0).expand(obs.shape[0], -1)
        logits = logits + self.risk_strength * risky.unsqueeze(0)
        if latent is not None:
            logits = logits + 0.3 * latent[:, :1] * risky.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if self.epsilon_mix > 0.0:
            uniform = torch.full_like(probs, 1.0 / float(self.n_actions))
            probs = (1.0 - self.epsilon_mix) * probs + self.epsilon_mix * uniform
        dist = Categorical(probs=probs)
        action = dist.sample()
        with torch.no_grad():
            hist = torch.bincount(action, minlength=self.n_actions).to(torch.float32)
            self._counts = self._counts + hist.cpu()
        return action.unsqueeze(-1), dist.log_prob(action)
