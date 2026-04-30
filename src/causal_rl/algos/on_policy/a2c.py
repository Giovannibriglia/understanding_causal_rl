from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from causal_rl.algos.base import BaseAlgorithm
from causal_rl.nets.mlp import MLP


class A2CPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.actor = MLP(obs_dim, n_actions)
        self.critic = MLP(obs_dim, 1)

    def action_dist(self, obs: Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: Tensor) -> Tensor:
        return cast(Tensor, self.critic(obs).squeeze(-1))


class A2C(BaseAlgorithm):
    kind = "on_policy"
    requires_continuous_action = False

    def __init__(
        self, obs_dim: int, n_actions: int, lr: float = 1e-3, ent_coef: float = 0.01
    ) -> None:
        self.policy = A2CPolicy(obs_dim, n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.ent_coef = ent_coef

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        dist = self.policy.action_dist(obs)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action.unsqueeze(-1), log_prob

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        action = batch["action"].view(-1)
        returns = batch["returns"]
        adv = batch["adv"]
        dist = self.policy.action_dist(obs)
        values = self.policy.value(obs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        policy_loss = -(log_prob * adv.detach()).mean()
        value_loss = 0.5 * ((returns - values) ** 2).mean()
        loss = policy_loss + value_loss - self.ent_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0).item()
        self.optimizer.step()
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "kl": 0.0,
            "grad_norm": float(grad_norm),
        }
