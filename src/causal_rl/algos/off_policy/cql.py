from __future__ import annotations

import copy

import torch
from torch import Tensor
from torch.distributions import Categorical

from causal_rl.algos.base import BaseAlgorithm
from causal_rl.nets.q_network import DiscreteQNetwork


class CQL(BaseAlgorithm):
    kind = "off_policy"
    requires_continuous_action = False

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 1.0,
        tau: float = 0.01,
    ) -> None:
        self.q = DiscreteQNetwork(obs_dim, n_actions)
        self.target_q = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau

    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]:
        qvals = self.q(obs)
        if deterministic:
            action = torch.argmax(qvals, dim=-1)
            log_prob = torch.zeros((obs.shape[0],), device=obs.device)
        else:
            dist = Categorical(logits=qvals)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.unsqueeze(-1), log_prob

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        action = batch["action"].view(-1).to(torch.long)
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        q_all = self.q(obs)
        q_sa = q_all.gather(1, action.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_q(next_obs).max(dim=-1).values
            target = reward + self.gamma * (1.0 - done) * next_q
        td_loss = ((q_sa - target) ** 2).mean()
        conservative = (torch.logsumexp(q_all, dim=-1) - q_sa).mean()
        loss = td_loss + self.alpha * conservative

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1.0).item()
        self.optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target_q.parameters(), strict=True):
                tp.mul_(1.0 - self.tau)
                tp.add_(self.tau * p)
        return {
            "td_loss": float(td_loss.item()),
            "policy_loss": float((self.alpha * conservative).item()),
            "value_loss": float(td_loss.item()),
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": float(grad_norm),
        }
