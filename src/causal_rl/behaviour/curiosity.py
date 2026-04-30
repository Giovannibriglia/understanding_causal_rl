from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class CuriosityExplorer(BiasedExplorer):
    family = "curiosity"

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        delta: float = 1.0,
        requires_latent: bool = False,
    ) -> None:
        self.n_actions = n_actions
        self.act_dim = act_dim
        self.delta = delta
        self.requires_latent = requires_latent
        self._counts = (
            torch.ones((self.n_actions,), dtype=torch.float32)
            if self.n_actions is not None
            else torch.empty((0,), dtype=torch.float32)
        )
        self._running_center = (
            torch.full((self.act_dim,), 0.5, dtype=torch.float32)
            if self.act_dim is not None
            else torch.empty((0,), dtype=torch.float32)
        )

    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.requires_latent and latent is None:
            msg = "Latent is required for confounded curiosity behaviour."
            raise ValueError(msg)
        if self.act_dim is not None:
            # Curiosity continuous behaviour:
            # prefer actions driven by novelty relative to an online running state center.
            state_slice = obs[:, : self.act_dim]
            center = self._running_center.to(obs.device).unsqueeze(0)
            novelty = (state_slice - center).abs()
            score = 1.4 * novelty - 0.35
            base = torch.sigmoid(self.delta * score)
            if latent is not None:
                base = torch.sigmoid(
                    torch.logit(base.clamp(1e-4, 1.0 - 1e-4)) + 0.10 * latent[:, :1]
                )
            action = (base + 0.03 * torch.randn_like(base)).clamp(0.0, 1.0)
            with torch.no_grad():
                batch_center = state_slice.detach().mean(dim=0).to(torch.float32).cpu()
                self._running_center = 0.9 * self._running_center + 0.1 * batch_center
            log_prob = torch.zeros((obs.shape[0],), device=obs.device)
            return action, log_prob
        if self.n_actions is None:
            msg = "Either n_actions or act_dim must be provided."
            raise ValueError(msg)
        counts = self._counts.to(obs.device)
        novelty = 1.0 / torch.sqrt(counts)
        logits = self.delta * novelty.unsqueeze(0).expand(obs.shape[0], -1)
        if latent is not None:
            logits = logits + 0.15 * latent[:, :1]
        dist = Categorical(logits=logits)
        action = dist.sample()
        with torch.no_grad():
            hist = torch.bincount(action, minlength=self.n_actions).to(torch.float32)
            self._counts = self._counts + hist.cpu()
        return action.unsqueeze(-1), dist.log_prob(action)
