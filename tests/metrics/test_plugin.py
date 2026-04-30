from __future__ import annotations

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv
from causal_rl.metrics.plugin import plugin_tv_gap


class _ToyEnv(CausalEnv):
    cell = 3
    obs_shape = (2,)
    act_shape = (1,)
    is_discrete_action = True
    horizon = 1
    gamma = 1.0

    def reset(self, seed: int | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        del seed
        return torch.zeros((4, 2)), {"latent_Z": torch.zeros((4, 1))}

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        del action
        b = 4
        return (
            torch.zeros((b, 2)),
            torch.zeros((b,)),
            torch.ones((b,), dtype=torch.bool),
            torch.zeros((b,), dtype=torch.bool),
            {"latent_Z": torch.zeros((b, 1))},
        )

    def do_reward(self, action: Tensor) -> Tensor:
        del action
        return torch.tensor([[0.5, 0.5]]).repeat(4, 1)

    def observed_reward_prob(self, action: Tensor) -> Tensor:
        del action
        return torch.tensor([[0.38, 0.62]]).repeat(4, 1)

    def do_transition(self, action: Tensor) -> Tensor:
        del action
        return torch.ones((4, 2))

    def close(self) -> None:
        return None


def test_plugin_convergence_to_known_delta() -> None:
    env = _ToyEnv()
    action = torch.zeros((4, 1), dtype=torch.long)
    est = plugin_tv_gap(env, action, pi_b_logprob=None).mean().item()
    assert abs(est - 0.12) <= 0.02
