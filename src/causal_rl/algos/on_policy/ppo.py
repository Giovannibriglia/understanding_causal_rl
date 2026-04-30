from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from causal_rl.algos.base import BaseAlgorithm
from causal_rl.nets.mlp import MLP


class PPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.actor = MLP(obs_dim, n_actions)
        self.critic = MLP(obs_dim, 1)

    def action_dist(self, obs: Tensor) -> Categorical:
        logits = self.actor(obs)
        return Categorical(logits=logits)

    def value(self, obs: Tensor) -> Tensor:
        return cast(Tensor, self.critic(obs).squeeze(-1))


class PPO(BaseAlgorithm):
    kind = "on_policy"
    requires_continuous_action = False

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
    ) -> None:
        self.policy = PPOPolicy(obs_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        dist = self.policy.action_dist(obs)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action.unsqueeze(-1), log_prob

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        action = batch["action"].view(-1)
        old_log_prob = batch["log_prob"]
        adv = batch["adv"]
        returns = batch["returns"]
        values_old = batch["value"]

        dist = self.policy.action_dist(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()

        values = self.policy.value(obs)
        v_clip = values_old + torch.clamp(values - values_old, -self.clip_eps, self.clip_eps)
        value_loss = 0.5 * torch.max((values - returns) ** 2, (v_clip - returns) ** 2).mean()

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0).item()
        self.optimizer.step()
        kl = (old_log_prob - log_prob).mean().abs().item()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "kl": kl,
            "grad_norm": float(grad_norm),
        }
