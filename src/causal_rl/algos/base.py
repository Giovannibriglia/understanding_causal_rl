from __future__ import annotations

import abc
from typing import Literal

from torch import Tensor


class BaseAlgorithm(abc.ABC):
    kind: Literal["on_policy", "off_policy"]
    requires_continuous_action: bool

    @abc.abstractmethod
    def select_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor]: ...

    @abc.abstractmethod
    def update(self, batch: dict[str, Tensor]) -> dict[str, float]: ...

    def n_informative_bounds(self) -> int:
        """Number of arms with tighter-than-trivial causal bounds. Override in subclasses."""
        return 0
