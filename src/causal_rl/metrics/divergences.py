from __future__ import annotations

import torch
from torch import Tensor


def _normalize(p: Tensor, eps: float = 1e-8) -> Tensor:
    p_safe = torch.clamp(p, min=eps)
    return p_safe / p_safe.sum(dim=-1, keepdim=True)


def total_variation(p_obs: Tensor, p_do: Tensor) -> Tensor:
    po = _normalize(p_obs)
    pd = _normalize(p_do)
    return 0.5 * torch.sum(torch.abs(po - pd), dim=-1)


def kl_divergence(p_obs: Tensor, p_do: Tensor) -> Tensor:
    support_violation = torch.any((p_obs > 0.0) & (p_do <= 0.0), dim=-1)
    po = _normalize(p_obs)
    pd = _normalize(p_do)
    kl = torch.sum(po * (torch.log(po) - torch.log(pd)), dim=-1)
    inf = torch.full_like(kl, float("inf"))
    return torch.where(support_violation, inf, kl)


def chi_squared(p_obs: Tensor, p_do: Tensor) -> Tensor:
    po = _normalize(p_obs)
    pd = _normalize(p_do)
    return torch.sum(((po - pd) ** 2) / pd, dim=-1)


def sup_log_ratio(p_obs: Tensor, p_do: Tensor) -> Tensor:
    po = _normalize(p_obs)
    pd = _normalize(p_do)
    return torch.max(torch.abs(torch.log(po) - torch.log(pd)), dim=-1).values
