from __future__ import annotations

import copy

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.algos.base import BaseAlgorithm
from causal_rl.nets.q_network import DiscreteQNetwork


class DQN(BaseAlgorithm):
    kind = "off_policy"
    requires_continuous_action = False

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        epsilon: float = 0.1,
        double: bool = True,
    ) -> None:
        self.q = DiscreteQNetwork(obs_dim, n_actions)
        self.target_q = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.double = double

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        qvals = self.q(obs)
        probs = torch.ones_like(qvals) * (self.epsilon / self.n_actions)
        greedy = torch.argmax(qvals, dim=-1, keepdim=True)
        probs.scatter_(1, greedy, 1.0 - self.epsilon + (self.epsilon / self.n_actions))
        dist = Categorical(probs=probs)
        action = greedy.squeeze(-1) if deterministic else dist.sample()
        return action.unsqueeze(-1), dist.log_prob(action)

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        action = batch["action"].view(-1).to(torch.long)
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        q_sa = self.q(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            if self.double:
                next_argmax = torch.argmax(self.q(next_obs), dim=-1, keepdim=True)
                next_q = self.target_q(next_obs).gather(1, next_argmax).squeeze(-1)
            else:
                next_q = self.target_q(next_obs).max(dim=-1).values
            target = reward + self.gamma * (1.0 - done) * next_q
        td_loss = torch.mean((q_sa - target) ** 2)
        self.optimizer.zero_grad()
        td_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0).item()
        self.optimizer.step()
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target_q.parameters(), strict=True):
                tp.mul_(1.0 - self.tau)
                tp.add_(self.tau * p)
        return {
            "td_loss": float(td_loss.item()),
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": float(grad_norm),
        }
