"""Phase 1.1 acceptance: ``bias_strength`` interpolation is continuous at 1.0.

Pre-v11 the ``_mix_with_uniform`` helper had an early-return at
``bias_strength >= 1.0`` that short-circuited the mixing coin-flip.
Combined with the empirical log-prob computed from coin draws, this
produced a *discontinuous* ``min_propensity`` / ``ess_ratio`` reading at
``bs=1.0`` — visible in the previous ``bias_sweep.pdf`` as a jump-back
on the rightmost bin.

The fix removes the early return so ``bs=1.0`` is the limit of the
mixing formula (every coin flip selects family).  This test exercises
the limit by sampling many trajectories at bs ∈ {0.95, 0.99, 1.0} and
checking that the empirical statistics are continuous.
"""

from __future__ import annotations

import math

import torch

from causal_rl.behaviour.reward_aligned import RewardAlignedExplorer
from causal_rl.metrics.coverage import effective_sample_size_ratio, min_propensity


def _empirical_log_probs(bias: float, n_calls: int = 2000, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    expl = RewardAlignedExplorer(n_actions=8, bias_strength=bias)
    obs = torch.zeros(n_calls, 5)
    latent = torch.bernoulli(torch.full((n_calls, 1), 0.5))
    _, lp = expl.select_action(obs, latent=latent)
    return lp


def test_min_propensity_continuous_at_bs_one() -> None:
    """``|mp(1.0) - mp(0.99)|`` is small relative to ``|mp(0.99) - mp(0.95)|``."""
    lp_95 = _empirical_log_probs(0.95)
    lp_99 = _empirical_log_probs(0.99)
    lp_100 = _empirical_log_probs(1.00)
    mp_95 = float(min_propensity(lp_95).item())
    mp_99 = float(min_propensity(lp_99).item())
    mp_100 = float(min_propensity(lp_100).item())
    diff_99_100 = abs(mp_100 - mp_99)
    diff_95_99 = abs(mp_99 - mp_95)
    # The 0.99→1.0 step should be at most twice the 0.95→0.99 step.  Pre-v11
    # it was an order of magnitude larger because of the early-return jump.
    assert diff_99_100 <= 2.0 * diff_95_99 + 1e-3, (
        f"min_propensity discontinuous at bs=1.0: "
        f"mp(0.95)={mp_95:.4f}, mp(0.99)={mp_99:.4f}, mp(1.0)={mp_100:.4f} "
        f"(99→100 step {diff_99_100:.4f} vs 95→99 step {diff_95_99:.4f})"
    )


def test_ess_ratio_continuous_at_bs_one() -> None:
    """ESS / N is also continuous at the bs=1.0 limit."""
    lp_99 = _empirical_log_probs(0.99)
    lp_100 = _empirical_log_probs(1.00)
    ess_99 = float(effective_sample_size_ratio(lp_99).item())
    ess_100 = float(effective_sample_size_ratio(lp_100).item())
    assert math.isfinite(ess_99) and math.isfinite(ess_100)
    # Both should be close to 1.0 for a 2000-sample biased exploration.
    assert abs(ess_100 - ess_99) < 0.05, (
        f"ESS ratio jumps at bs=1.0: ess(0.99)={ess_99:.3f}, ess(1.0)={ess_100:.3f}"
    )
