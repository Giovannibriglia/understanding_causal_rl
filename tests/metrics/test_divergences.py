from __future__ import annotations

import torch

from causal_rl.metrics.divergences import chi_squared, kl_divergence, sup_log_ratio, total_variation


def test_divergences_zero_for_equal_distributions() -> None:
    p = torch.tensor([[0.3, 0.7], [0.5, 0.5]], dtype=torch.float32)
    assert torch.allclose(total_variation(p, p), torch.zeros(2), atol=1e-6)
    assert torch.allclose(kl_divergence(p, p), torch.zeros(2), atol=1e-6)
    assert torch.allclose(chi_squared(p, p), torch.zeros(2), atol=1e-6)
    assert torch.allclose(sup_log_ratio(p, p), torch.zeros(2), atol=1e-6)


def test_orthogonal_support_extremes() -> None:
    p = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    q = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    tv = total_variation(p, q)
    kl = kl_divergence(p, q)
    assert torch.allclose(tv, torch.tensor([1.0]), atol=1e-6)
    assert torch.isinf(kl).all()
