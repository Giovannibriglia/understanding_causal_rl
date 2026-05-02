from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.behaviour.base import BiasedExplorer


class RewardMisalignedExplorer(BiasedExplorer):
    family = "reward_misaligned"
    depends_on_u = True

    def __init__(
        self,
        n_actions: int | None = None,
        act_dim: int | None = None,
        beta: float = 1.0,
        bias_strength: float = 1.0,
        epsilon_mix: float = 0.05,
        requires_latent: bool = False,
    ) -> None:
        super().__init__(bias_strength=bias_strength)
        self.n_actions = n_actions
        self.act_dim = act_dim
        self.beta = beta
        self.epsilon_mix = epsilon_mix
        self.requires_latent = requires_latent
        if self.n_actions is not None:
            self._weights = torch.linspace(-0.5, 0.5, steps=self.n_actions)
            risk = torch.linspace(0.0, 1.0, steps=self.n_actions)
            self._risk_profile = risk
        else:
            self._weights = torch.empty((0,), dtype=torch.float32)
            self._risk_profile = torch.empty((0,), dtype=torch.float32)

    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.requires_latent and latent is None:
            msg = "Latent is required for confounded reward-misaligned behaviour."
            raise ValueError(msg)
        if self.act_dim is not None:
            # Reward-misaligned continuous behaviour:
            # anti-align with state-implied treatment and inject explicit risky bias.
            state_slice = obs[:, : self.act_dim]
            state_context = obs.mean(dim=-1, keepdim=True)
            anti_score = -(0.80 * state_slice + 0.20 * state_context)
            risky_push = 0.55 * self.bias_strength
            base = torch.sigmoid(self.beta * anti_score + risky_push)
            if latent is not None:
                base = torch.sigmoid(
                    torch.logit(base.clamp(1e-4, 1.0 - 1e-4)) + 0.10 * latent[:, :1]
                )
            action = (base + 0.03 * torch.randn_like(base)).clamp(0.0, 1.0)
            log_prob = torch.zeros((obs.shape[0],), device=obs.device)
            return self._mix_with_uniform(
                obs, action, log_prob, n_actions=self.n_actions, act_dim=self.act_dim
            )
        if self.n_actions is None:
            msg = "Either n_actions or act_dim must be provided."
            raise ValueError(msg)
        state_score = obs.mean(dim=-1, keepdim=True)
        weights = self._weights.to(obs.device).unsqueeze(0)
        risk = self._risk_profile.to(obs.device).unsqueeze(0)
        scores = -(state_score * weights) + self.bias_strength * risk
        if latent is not None:
            scores = scores + 0.5 * latent[:, :1] * risk
        logits = self.beta * scores
        probs = torch.softmax(logits, dim=-1)
        if self.epsilon_mix > 0.0:
            uniform = torch.full_like(probs, 1.0 / float(self.n_actions))
            probs = (1.0 - self.epsilon_mix) * probs + self.epsilon_mix * uniform
        dist = Categorical(probs=probs)
        action = dist.sample()
        return self._mix_with_uniform(
            obs, action.unsqueeze(-1), dist.log_prob(action), n_actions=self.n_actions
        )
