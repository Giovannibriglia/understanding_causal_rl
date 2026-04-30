from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ReplayBatch:
    obs: Tensor
    action: Tensor
    reward: Tensor
    next_obs: Tensor
    done: Tensor
    behaviour_logprob: Tensor
    latent: Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        discrete_action: bool,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.discrete_action = discrete_action
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        action_dtype = torch.long if discrete_action else torch.float32
        self.action = torch.zeros((capacity, act_dim), dtype=action_dtype, device=device)
        self.reward = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.done = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.behaviour_logprob = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.latent = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self._size = 0
        self._ptr = 0

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: Tensor,
        action: Tensor,
        reward: Tensor,
        next_obs: Tensor,
        done: Tensor,
        behaviour_logprob: Tensor | None,
        latent: Tensor | None,
    ) -> None:
        n = obs.shape[0]
        for i in range(n):
            idx = self._ptr
            self.obs[idx] = obs[i]
            self.action[idx] = action[i]
            self.reward[idx] = reward[i]
            self.next_obs[idx] = next_obs[i]
            self.done[idx] = done[i]
            self.behaviour_logprob[idx] = (
                behaviour_logprob[i]
                if behaviour_logprob is not None
                else torch.tensor(0.0, device=self.device)
            )
            self.latent[idx] = (
                latent[i] if latent is not None else torch.tensor([0.0], device=self.device)
            )
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self._size == 0:
            msg = "Cannot sample from empty buffer."
            raise RuntimeError(msg)
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return ReplayBatch(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_obs=self.next_obs[idx],
            done=self.done[idx],
            behaviour_logprob=self.behaviour_logprob[idx],
            latent=self.latent[idx],
        )
