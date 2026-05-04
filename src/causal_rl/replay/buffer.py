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
        # ``true_behaviour_logprob`` is always populated and never exposed to
        # the algorithm — the runner uses it for π_b-recovery diagnostics
        # (e.g. ``pi_b_recovery_kl``) in cells where ``pi_b_known=False``.
        # NaN-initialised to distinguish "not recorded" from "recorded zero".
        self.true_behaviour_logprob = torch.full(
            (capacity,), float("nan"), dtype=torch.float32, device=device
        )
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
        true_behaviour_logprob: Tensor | None = None,
    ) -> None:
        """Vectorised batch insert into the circular buffer."""
        n = obs.shape[0]
        # Build contiguous index array (may wrap around)
        indices = torch.arange(self._ptr, self._ptr + n, device=self.device) % self.capacity
        self.obs[indices] = obs.to(self.device)
        self.action[indices] = action.to(self.device)
        self.reward[indices] = reward.view(n).to(self.device)
        self.next_obs[indices] = next_obs.to(self.device)
        self.done[indices] = done.view(n).to(self.device)
        if behaviour_logprob is not None:
            self.behaviour_logprob[indices] = behaviour_logprob.view(n).to(self.device)
        else:
            self.behaviour_logprob[indices] = 0.0
        if true_behaviour_logprob is not None:
            self.true_behaviour_logprob[indices] = true_behaviour_logprob.view(n).to(
                self.device
            )
        if latent is not None:
            self.latent[indices] = latent.view(n, -1).to(self.device)
        else:
            self.latent[indices] = 0.0
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

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
