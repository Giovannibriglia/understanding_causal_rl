from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from causal_rl.nets.mlp import MLP


class DiscreteQNetwork(nn.Module):
    def __init__(
        self, obs_dim: int, n_actions: int, hidden_dims: tuple[int, ...] = (128, 128)
    ) -> None:
        super().__init__()
        self.model = MLP(obs_dim, n_actions, hidden_dims=hidden_dims, activation=nn.ReLU)

    def forward(self, obs: Tensor) -> Tensor:
        return cast(Tensor, self.model(obs))


class ContinuousQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.model = MLP(obs_dim + act_dim, out_dim, hidden_dims=hidden_dims, activation=nn.ReLU)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return cast(Tensor, self.model(torch.cat([obs, action], dim=-1)))
