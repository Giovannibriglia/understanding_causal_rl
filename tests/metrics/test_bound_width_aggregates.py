"""v14 contract: bound_width aggregates.

``bound_width_mean`` is mathematically constant at ``1 - 1/n_actions``
for any p_a summing to 1 — it encodes nothing about the run.
``bound_width_max`` (rarest-played arm) and ``bound_width_std`` (spread
across arms) are the two informative aggregates that ``static_diagnostics``
plots in v14.
"""

from __future__ import annotations

import torch

from causal_rl.identification.bounds import natural_bounds


def _widths(actions: torch.Tensor, n_actions: int = 8) -> list[float]:
    rewards = torch.zeros_like(actions, dtype=torch.float)
    bounds = natural_bounds(actions, rewards.float(), n_actions, n_bootstrap=10)
    return [bounds[a].upper - bounds[a].lower for a in range(n_actions)]


def test_bound_width_std_is_zero_under_uniform_play() -> None:
    """Uniform play → all p_a equal → all widths equal → std = 0."""
    actions = torch.arange(8).repeat(100)  # 800 transitions, 100 per arm
    widths = _widths(actions)
    assert max(widths) - min(widths) < 1e-6
    std = float(
        torch.tensor(widths, dtype=torch.float32).std(unbiased=False).item()
    )
    assert std < 1e-6


def test_bound_width_std_grows_with_bias() -> None:
    """Concentrated play → widths spread out → std > 0."""
    # 80% of transitions on arm 0, 20% spread across arms 1-7.
    biased = torch.cat(
        [
            torch.zeros(800, dtype=torch.long),
            torch.arange(1, 8).repeat(28)[:200],
        ]
    )
    widths = _widths(biased)
    std = float(
        torch.tensor(widths, dtype=torch.float32).std(unbiased=False).item()
    )
    # Visibly above noise.  With p_0=0.8 and p_other≈0.029 each, the
    # widths are 0.2 and 0.97 respectively — std easily > 0.05.
    assert std > 0.05


def test_bound_width_max_is_one_minus_min_p_a() -> None:
    """The max width corresponds to the arm with the smallest p_a:
    ``max_a (r_a − l_a) = max_a (1 − p_a) = 1 − min_a p_a``."""
    # Arm 7 played only twice in 800 transitions.
    actions = torch.cat(
        [
            torch.arange(7).repeat(114),  # 7 arms × 114 = 798
            torch.tensor([7, 7], dtype=torch.long),  # arm 7 played twice
        ]
    )
    widths = _widths(actions, n_actions=8)
    n = float(actions.numel())
    expected_min_p = 2.0 / n
    expected_max_width = 1.0 - expected_min_p
    assert abs(max(widths) - expected_max_width) < 1e-3


def test_bound_width_mean_is_algebraic_identity() -> None:
    """``mean_a (1 − p_a) = 1 − 1/n_actions`` for any p_a summing to 1.

    This is the structural reason ``bound_width_mean`` was excluded from
    the v14 ``static_diagnostics`` figure — it cannot vary across runs.
    """
    n_actions = 8
    expected = 1.0 - 1.0 / float(n_actions)
    # Three arbitrary action distributions; all should give the same mean.
    for actions in (
        torch.arange(8).repeat(100),
        torch.cat(
            [
                torch.zeros(700, dtype=torch.long),
                torch.arange(1, 8).repeat(15)[:100],
            ]
        ),
        torch.tensor([0] * 200 + [3] * 600, dtype=torch.long),
    ):
        widths = _widths(actions, n_actions=n_actions)
        mean = sum(widths) / len(widths)
        assert abs(mean - expected) < 1e-3, (
            f"bound_width_mean = {mean:.6f}, expected {expected:.6f}; "
            "this should be an algebraic identity"
        )
