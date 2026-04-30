from __future__ import annotations

import abc
from typing import Literal

from torch import Tensor


class BiasedExplorer(abc.ABC):
    family: Literal["baseline", "reward_aligned", "information", "curiosity", "reward_misaligned"]
    requires_latent: bool

    @abc.abstractmethod
    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
