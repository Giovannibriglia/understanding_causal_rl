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

    def sample_interventional(self, action: Tensor, n: int) -> Tensor:
        """Draw n reward samples per env from P(R | do(A=a), current observable state).

        For discrete envs returns shape (n_envs, n) with values in {-1, +1}.
        For continuous envs returns shape (n_envs, n) of scalar rewards.
        Subclasses should override to provide env-specific sampling.
        """
        raise NotImplementedError

    def sample_observational(self, action: Tensor, n: int) -> Tensor:
        """Draw n reward samples per env from P(R | A=a, current observable state).

        The observational distribution may differ from the interventional one when
        the behaviour policy is correlated with unobserved confounders.  The degree
        of divergence is controlled by the env's ``alpha_conf`` parameter (if any).

        For discrete envs returns shape (n_envs, n) with values in {-1, +1}.
        For continuous envs returns shape (n_envs, n) of scalar rewards.
        Subclasses should override to provide env-specific sampling.
        """
        raise NotImplementedError
