from __future__ import annotations

import abc

from torch import Tensor


class CausalEnv(abc.ABC):
    cell: int
    obs_shape: tuple[int, ...]
    act_shape: tuple[int, ...]
    is_discrete_action: bool
    horizon: int
    gamma: float

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> tuple[Tensor, dict[str, Tensor]]: ...

    @abc.abstractmethod
    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]: ...

    @abc.abstractmethod
    def do_reward(self, action: Tensor) -> Tensor: ...

    @abc.abstractmethod
    def do_transition(self, action: Tensor) -> Tensor: ...

    @abc.abstractmethod
    def close(self) -> None: ...
