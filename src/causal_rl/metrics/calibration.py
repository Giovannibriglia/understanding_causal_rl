from __future__ import annotations

import torch
from torch import Tensor


def expected_calibration_error(
    predicted_probs: Tensor,
    outcomes: Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) for a binary outcome.

    Bins observations by predicted probability of the taken action, then
    measures the weighted-average gap between mean confidence and fraction
    of positive outcomes in each bin.

    Args:
        predicted_probs: (N,) predicted probability of taken action in [0, 1].
        outcomes: (N,) binary outcomes in {0, 1}.
        n_bins: Number of equal-width bins over [0, 1].

    Returns:
        ECE as a float in [0, 1].
    """
    n = predicted_probs.shape[0]
    if n == 0:
        return 0.0
    outcomes_f = outcomes.float()
    edges = torch.linspace(0.0, 1.0 + 1e-8, n_bins + 1, device=predicted_probs.device)
    ece = 0.0
    for i in range(n_bins):
        lo = float(edges[i].item())
        hi = float(edges[i + 1].item())
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        n_in_bin = int(mask.sum().item())
        if n_in_bin == 0:
            continue
        mean_conf = float(predicted_probs[mask].mean().item())
        frac_pos = float(outcomes_f[mask].mean().item())
        ece += (n_in_bin / n) * abs(mean_conf - frac_pos)
    return ece
