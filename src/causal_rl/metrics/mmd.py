from __future__ import annotations

import torch
from torch import Tensor


def _pairwise_sqdist(x: Tensor, y: Tensor) -> Tensor:
    x2 = (x**2).sum(dim=-1, keepdim=True)
    y2 = (y**2).sum(dim=-1, keepdim=True).transpose(0, 1)
    return torch.clamp(x2 + y2 - 2.0 * x @ y.transpose(0, 1), min=0.0)


def mmd2(x: Tensor, y: Tensor) -> Tensor:
    dxx = _pairwise_sqdist(x, x)
    dyy = _pairwise_sqdist(y, y)
    dxy = _pairwise_sqdist(x, y)
    flat = dxy.flatten()
    bw = torch.median(flat[flat > 0]) if torch.any(flat > 0) else torch.tensor(1.0, device=x.device)
    bw = torch.clamp(bw, min=1e-6)
    kxx = torch.exp(-dxx / bw)
    kyy = torch.exp(-dyy / bw)
    kxy = torch.exp(-dxy / bw)
    return kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
