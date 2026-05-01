from __future__ import annotations

import torch

from causal_rl.metrics.coverage import expected_calibration_error


def test_expected_calibration_error_perfect_classifier_near_zero() -> None:
    n = 100
    actions = torch.randint(0, 2, (n,))
    probs = torch.zeros((n, 2), dtype=torch.float32)
    probs[torch.arange(n), actions] = 0.999
    probs[torch.arange(n), 1 - actions] = 0.001
    ece = expected_calibration_error(torch.log(probs), actions, n_bins=10)
    assert float(ece.item()) < 0.05


def test_expected_calibration_error_uniform_on_biased_data() -> None:
    n = 200
    actions = torch.cat([torch.ones(160, dtype=torch.long), torch.zeros(40, dtype=torch.long)])
    actions = actions[torch.randperm(n)]
    probs = torch.full((n, 2), 0.5, dtype=torch.float32)
    ece = expected_calibration_error(torch.log(probs), actions, n_bins=10)
    assert abs(float(ece.item()) - 0.3) < 0.1
