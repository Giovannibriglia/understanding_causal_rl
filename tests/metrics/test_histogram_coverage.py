"""Unit tests for the v12 histogram-based coverage diagnostics.

These metrics complement the existing log-prob-based ``min_propensity`` /
``effective_sample_size_ratio`` by reading the empirical action histogram
directly.  They reveal masked-coverage regimes that the per-row log_prob
diagnostics can't see — see ``metrics/coverage.py`` docstring.
"""

from __future__ import annotations

import pytest
import torch

from causal_rl.metrics.coverage import ess_histogram, min_action_freq


def test_min_action_freq_zero_when_action_unobserved() -> None:
    """If action 0 never appears, ``min_action_freq`` is exactly 0."""
    actions = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    assert float(min_action_freq(actions, n_actions=8).item()) == 0.0


def test_min_action_freq_uniform() -> None:
    """All 8 actions equally — min freq = 1/8."""
    actions = torch.arange(8).repeat(100)  # 800 transitions, 100 of each
    expected = 1.0 / 8.0
    got = float(min_action_freq(actions, n_actions=8).item())
    assert abs(got - expected) < 1e-6


def test_min_action_freq_empty_buffer_is_nan() -> None:
    actions = torch.empty((0,), dtype=torch.long)
    val = float(min_action_freq(actions, n_actions=8).item())
    assert val != val  # NaN


def test_ess_histogram_uniform_is_one() -> None:
    """Perfectly uniform → ESS ratio = 1.0."""
    actions = torch.arange(8).repeat(100)
    got = float(ess_histogram(actions, n_actions=8).item())
    assert got == pytest.approx(1.0, abs=1e-6)


def test_ess_histogram_falls_with_masked_actions() -> None:
    """4 of 8 arms unobserved → ESS/N at most 0.5."""
    actions = torch.tensor([4, 5, 6, 7, 4, 5, 6, 7] * 100)
    got = float(ess_histogram(actions, n_actions=8).item())
    assert got <= 0.5 + 1e-6


def test_ess_histogram_single_point_mass() -> None:
    """All transitions on one arm → ESS/N = 1/n_actions."""
    actions = torch.zeros(800, dtype=torch.long)
    got = float(ess_histogram(actions, n_actions=8).item())
    expected = 1.0 / 8.0
    assert abs(got - expected) < 1e-6


def test_ess_histogram_floor_at_one_over_n_actions() -> None:
    """Two arms touched, six dead — ESS/N still bounded below by 1/n_actions."""
    actions = torch.tensor([0] * 700 + [1] * 100)
    got = float(ess_histogram(actions, n_actions=8).item())
    assert got >= 1.0 / 8.0 - 1e-6


def test_ess_histogram_capped_at_one() -> None:
    """Even with a large buffer that's perfectly uniform, the cap holds."""
    actions = torch.arange(8).repeat(10_000)
    got = float(ess_histogram(actions, n_actions=8).item())
    assert got <= 1.0 + 1e-9
