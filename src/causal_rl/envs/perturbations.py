"""Environment perturbation schedules for distribution-shift evaluation (Phase E).

Two perturbation axes per environment on a 5-level diagonal schedule:
  Tabular sepsis:
    eps_T  ∈ {0, 0.1, 0.2, 0.35, 0.5} — rotation on latent transition tensor.
    eps_R  ∈ {0, 0.05, 0.1, 0.2, 0.3} — reward shift toward 0.5.
  Continuous ward:
    eps_target   ∈ {0, 0.1, 0.2, 0.35, 0.5} — additive shift on target state.
    eps_dynamics ∈ {0, 0.05, 0.1, 0.2, 0.3} — scale on dt and sigma.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PerturbationSpec:
    eps_T: float = 0.0  # noqa: N815
    eps_R: float = 0.0  # noqa: N815
    eps_target: float = 0.0
    eps_dynamics: float = 0.0


# The 5-point diagonal schedule
TABULAR_DIAGONAL: list[PerturbationSpec] = [
    PerturbationSpec(eps_T=0.0, eps_R=0.0),
    PerturbationSpec(eps_T=0.1, eps_R=0.05),
    PerturbationSpec(eps_T=0.2, eps_R=0.1),
    PerturbationSpec(eps_T=0.35, eps_R=0.2),
    PerturbationSpec(eps_T=0.5, eps_R=0.3),
]

CONTINUOUS_DIAGONAL: list[PerturbationSpec] = [
    PerturbationSpec(eps_target=0.0, eps_dynamics=0.0),
    PerturbationSpec(eps_target=0.1, eps_dynamics=0.05),
    PerturbationSpec(eps_target=0.2, eps_dynamics=0.1),
    PerturbationSpec(eps_target=0.35, eps_dynamics=0.2),
    PerturbationSpec(eps_target=0.5, eps_dynamics=0.3),
]


def perturb_tabular_transition(
    transition: Tensor,
    eps_T: float,  # noqa: N803
    seed: int = 0,
) -> Tensor:
    """Apply a random unitary rotation (restricted to a 5-dim subspace) to the transition tensor.

    transition: (S, A, S) probability tensor.
    Returns a new (S, A, S) tensor with rows re-normalised.
    """
    if eps_T <= 0.0:
        return transition
    torch.manual_seed(seed)
    s, a, sp = transition.shape
    out = transition.clone()
    for ai in range(a):
        block = out[:, ai, :]  # (S, S)
        dim = min(5, s)
        indices = torch.randperm(s, generator=torch.Generator().manual_seed(seed + ai))[:dim]
        submat = block[indices][:, indices]
        noise = eps_T * torch.randn_like(submat)
        noise = noise - noise.mean(dim=-1, keepdim=True)
        submat = (submat + noise).clamp(0.0)
        row_sum = submat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        submat = submat / row_sum
        block[indices.unsqueeze(1), indices.unsqueeze(0)] = submat
        out[:, ai, :] = block
    row_sum = out.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return out / row_sum


def perturb_tabular_rewards(
    reward_probs: Tensor,
    eps_R: float,  # noqa: N803
) -> Tensor:
    """Shift reward probabilities toward 0.5 by eps_R.

    reward_probs: (S, A, 2) where [:,  :, 1] = P(+1 reward).
    Returns new tensor with p_pos shifted: p' = (1 - eps_R) * p + eps_R * 0.5.
    """
    if eps_R <= 0.0:
        return reward_probs
    out = reward_probs.clone()
    p_pos = out[:, :, 1]
    p_pos_new = (1.0 - eps_R) * p_pos + eps_R * 0.5
    out[:, :, 1] = p_pos_new
    out[:, :, 0] = 1.0 - p_pos_new
    return out
