from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from causal_rl.algos.base import BaseAlgorithm
from causal_rl.nets.mlp import MLP
from causal_rl.nets.q_network import ContinuousQNetwork


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.model = MLP(obs_dim, act_dim)

    def forward(self, obs: Tensor) -> Tensor:
        return torch.sigmoid(self.model(obs))


class DDPG(BaseAlgorithm):
    kind = "off_policy"
    requires_continuous_action = True

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        noise_std: float = 0.1,
    ) -> None:
        self.actor = DeterministicPolicy(obs_dim, act_dim)
        self.critic = ContinuousQNetwork(obs_dim, act_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        action = self.actor(obs)
        if not deterministic:
            action = (action + self.noise_std * torch.randn_like(action)).clamp(0.0, 1.0)
        log_prob = torch.zeros((obs.shape[0],), device=obs.device)
        return action, log_prob

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        obs = batch["obs"].detach()
        action = batch["action"].detach()
        reward = batch["reward"].detach()
        next_obs = batch["next_obs"].detach()
        done = batch["done"].detach()

        with torch.no_grad():
            next_action = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, next_action).squeeze(-1)
            y = reward + self.gamma * (1.0 - done) * target_q
        q = self.critic(obs, action).squeeze(-1)
        critic_loss = ((q - y) ** 2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Freeze critic parameters while optimizing the actor to avoid
        # accidental in-place versioning conflicts across two backward passes.
        for p in self.critic.parameters():
            p.requires_grad_(False)
        actor_action = self.actor(obs)
        actor_loss = -self.critic(obs, actor_action).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0).item()
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.target_actor.parameters(), strict=True):
                tp.mul_(1.0 - self.tau)
                tp.add_(self.tau * p)
            for p, tp in zip(
                self.critic.parameters(), self.target_critic.parameters(), strict=True
            ):
                tp.mul_(1.0 - self.tau)
                tp.add_(self.tau * p)

        return {
            "td_loss": float(critic_loss.item()),
            "policy_loss": float(actor_loss.item()),
            "value_loss": float(critic_loss.item()),
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": float(grad_norm),
        }
