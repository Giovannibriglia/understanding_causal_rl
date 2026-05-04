"""Backdoor adjustment residual.

For a confounder set ``Z`` declared as observed, the back-door adjustment
formula gives the interventional mean reward as

    E_BD[R | A=a]  =  Σ_z  P̂(Z=z) · Ê[R | A=a, Z=z]

while the *naive* observational estimator is just

    E_naive[R | A=a]  =  Ê[R | A=a].

When ``Z`` actually satisfies the back-door criterion (cells with
``expose_z=True`` and no other unobserved confounding) the two converge
as data grows.  Otherwise their gap measures the residual confounding
that ``Z`` failed to block.

References:
    Pearl 2009, *Causality* (2nd ed.), §3.3 "back-door criterion".
    Bareinboim & Pearl 2016, *PNAS* — limits of identifiability.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def backdoor_residual_per_arm(
    actions: Tensor,
    rewards: Tensor,
    z_observed: Tensor | None,
    n_actions: int,
) -> dict[int, float]:
    """E[R|A=a] − Σ_z P̂(z) Ê[R | A=a, Z=z], per arm.

    Args:
        actions: ``(N,)`` integer actions in ``[0, n_actions)``.
        rewards: ``(N,)`` float rewards (any range).
        z_observed: ``(N,)`` integer observed-confounder values, or ``None``
            when ``Z`` is hidden by the cell.  ``None`` is the signal
            "back-door is impossible because the adjustment set is empty".
        n_actions: Total number of arms.

    Returns:
        Mapping ``arm -> residual``.  When ``z_observed`` is ``None`` every
        residual is ``NaN``.  When an arm was never selected the residual is
        ``NaN`` for that arm.
    """
    if z_observed is None:
        return {a: float("nan") for a in range(n_actions)}
    acts = actions.view(-1).long()
    rews = rewards.view(-1).float()
    z = z_observed.view(-1).long()
    if acts.numel() == 0 or z.numel() != acts.numel():
        return {a: float("nan") for a in range(n_actions)}
    # Marginal P̂(Z=z) over the full sample.
    z_values = torch.unique(z).tolist()
    p_z = {int(zv): float((z == zv).float().mean().item()) for zv in z_values}
    out: dict[int, float] = {}
    for a in range(n_actions):
        mask_a = acts == a
        if not bool(mask_a.any()):
            out[a] = float("nan")
            continue
        e_naive = float(rews[mask_a].mean().item())
        e_bd = 0.0
        weight_used = 0.0
        for zv in z_values:
            mask_az = mask_a & (z == zv)
            if not bool(mask_az.any()):
                # No (a, z) cell observed — skip and renormalise.
                continue
            e_az = float(rews[mask_az].mean().item())
            e_bd += p_z[int(zv)] * e_az
            weight_used += p_z[int(zv)]
        if weight_used <= 0.0:
            out[a] = float("nan")
            continue
        e_bd_normalised = e_bd / weight_used
        out[a] = float(e_naive - e_bd_normalised)
    return out


def backdoor_residual_summary(
    actions: Tensor,
    rewards: Tensor,
    z_observed: Tensor | None,
    n_actions: int,
) -> dict[str, float]:
    """Summary statistics for :func:`backdoor_residual_per_arm`.

    Returns a dict with keys ``backdoor_residual_mean`` and
    ``backdoor_residual_max`` (both NaN when Z is hidden or no per-arm
    residual could be computed).
    """
    per_arm = backdoor_residual_per_arm(actions, rewards, z_observed, n_actions)
    finite = [abs(v) for v in per_arm.values() if not math.isnan(v)]
    if not finite:
        return {
            "backdoor_residual_mean": float("nan"),
            "backdoor_residual_max": float("nan"),
        }
    return {
        "backdoor_residual_mean": float(sum(finite) / len(finite)),
        "backdoor_residual_max": float(max(finite)),
    }


__all__ = ["backdoor_residual_per_arm", "backdoor_residual_summary"]
