from __future__ import annotations

import abc
from typing import Literal

from torch import Tensor


class BiasedExplorer(abc.ABC):
    family: Literal["baseline", "reward_aligned", "information", "curiosity", "reward_misaligned"]
    requires_latent: bool
    # True when this policy conditions on U (the hidden confounder), creating a
    # spurious A↔R correlation that affects identifiability in pi_b_unknown cells.
    depends_on_u: bool

    @abc.abstractmethod
    def select_action(self, obs: Tensor, latent: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
