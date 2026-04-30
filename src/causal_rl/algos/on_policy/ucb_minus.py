from __future__ import annotations

import torch

from causal_rl.algos.on_policy.ucb import UCB


class UCBMinus(UCB):
    """UCB initialised with observational arm means as a warm start.

    If ``observational_means`` is provided the empirical means are seeded
    with those values and each arm is treated as if it has been observed once,
    giving the exploration bonus a head-start that reflects prior (potentially
    confounded) data.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        observational_means: list[float] | None = None,
    ) -> None:
        super().__init__(obs_dim, n_actions)
        if observational_means is not None:
            self._means = torch.tensor(observational_means, dtype=torch.float32)
            self._counts = torch.ones(n_actions, dtype=torch.float32)
            self._t = n_actions
