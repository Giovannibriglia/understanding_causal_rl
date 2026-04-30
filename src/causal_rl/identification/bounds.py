from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BoundResult:
    """Point estimates and bootstrap CIs for one arm's natural bounds."""

    lower: float
    upper: float
    lower_ci_lo: float
    lower_ci_hi: float
    upper_ci_lo: float
    upper_ci_hi: float
    p_a: float  # P(A = a)


def natural_bounds(
    actions: Tensor,
    rewards_01: Tensor,
    n_actions: int,
    n_bootstrap: int = 200,
) -> dict[int, BoundResult]:
    """Per-arm Bareinboim Thm 9.2.2 natural bounds with bootstrap CIs.

    Bounds for arm *a*::

        l_a = E[Y | A=a] * P(A=a)
        r_a = E[Y | A=a] * P(A=a) + (1 - P(A=a))

    .. note::
        ``rewards_01`` must be in [0, 1].  For rewards in {-1, +1}, scale
        via ``(r + 1) / 2``.

    Args:
        actions: ``(N,)`` integer actions in ``[0, n_actions)``.
        rewards_01: ``(N,)`` float rewards in ``[0, 1]``.
        n_actions: Total number of arms.
        n_bootstrap: Number of bootstrap replicates used for CIs.

    Returns:
        Mapping from arm index to :class:`BoundResult`.
    """
    n = actions.shape[0]

    def _compute_bounds(acts: Tensor, rews: Tensor) -> dict[int, tuple[float, float, float]]:
        """Return {arm: (lower, upper, p_a)} for one sample."""
        result: dict[int, tuple[float, float, float]] = {}
        total = acts.shape[0]
        for a in range(n_actions):
            mask = acts == a
            count = mask.sum().item()
            p_a = count / total
            ey_given_a = rews[mask].mean().item() if count > 0 else 0.0
            lower = ey_given_a * p_a
            upper = ey_given_a * p_a + (1.0 - p_a)
            result[a] = (lower, upper, float(p_a))
        return result

    # Point estimates
    point = _compute_bounds(actions, rewards_01)

    # Bootstrap replicates
    boot_lower: dict[int, list[float]] = {a: [] for a in range(n_actions)}
    boot_upper: dict[int, list[float]] = {a: [] for a in range(n_actions)}

    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,))
        boot_acts = actions[idx]
        boot_rews = rewards_01[idx]
        rep = _compute_bounds(boot_acts, boot_rews)
        for a in range(n_actions):
            lo, hi, _ = rep[a]
            boot_lower[a].append(lo)
            boot_upper[a].append(hi)

    results: dict[int, BoundResult] = {}
    for a in range(n_actions):
        lo_pt, hi_pt, p_a_pt = point[a]
        lo_arr = torch.tensor(boot_lower[a])
        hi_arr = torch.tensor(boot_upper[a])
        results[a] = BoundResult(
            lower=lo_pt,
            upper=hi_pt,
            lower_ci_lo=float(torch.quantile(lo_arr, 0.025).item()),
            lower_ci_hi=float(torch.quantile(lo_arr, 0.975).item()),
            upper_ci_lo=float(torch.quantile(hi_arr, 0.025).item()),
            upper_ci_hi=float(torch.quantile(hi_arr, 0.975).item()),
            p_a=p_a_pt,
        )

    return results
