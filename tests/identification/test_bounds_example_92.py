from __future__ import annotations

import pytest
import torch

from causal_rl.identification.bounds import BoundResult, natural_bounds

# ---------------------------------------------------------------------------
# Bareinboim Example 9.2  (simplified observational distribution)
# ---------------------------------------------------------------------------
#
# Two arms: A ∈ {0, 1}
#   P(A=0) = 0.4,  P(A=1) = 0.6
#   E[Y | A=0]     = 0.0
#   E[Y | A=1]     = 0.5 - delta          (delta = 0.1 → 0.4)
#
# True interventional values (Bareinboim):
#   E[Y | do(A=0)] = 0.4
#   E[Y | do(A=1)] = 0.5 - 1.25 * delta   (delta = 0.1 → 0.375)
#
# Natural bounds for arm a:
#   l_a = E[Y | A=a] * P(A=a)
#   r_a = E[Y | A=a] * P(A=a) + (1 - P(A=a))
# ---------------------------------------------------------------------------

DELTA: float = 0.1
N_OBS: int = 50_000
TOL: float = 0.05  # loose tolerance for finite-sample checks


def _generate_observations(
    n: int, delta: float, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic observations matching Example 9.2.

    Arm 0: selected 40% of the time, reward ~ Bernoulli(0.0).
    Arm 1: selected 60% of the time, reward ~ Bernoulli(0.5 - delta).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Arm assignments
    actions = torch.bernoulli(torch.full((n,), 0.6), generator=rng).long()  # P(A=1) = 0.6

    # Rewards
    p_reward_0 = 0.0
    p_reward_1 = 0.5 - delta
    rewards = torch.where(
        actions == 0,
        torch.bernoulli(torch.full((n,), p_reward_0), generator=rng),
        torch.bernoulli(torch.full((n,), p_reward_1), generator=rng),
    )

    return actions, rewards


# ---------------------------------------------------------------------------
# Expected analytic values
# ---------------------------------------------------------------------------

# P(A=a)
P_A0 = 0.4
P_A1 = 0.6

# E[Y | A=a]
EY_GIVEN_A0 = 0.0
EY_GIVEN_A1 = 0.5 - DELTA  # = 0.4

# Natural bounds (point values)
L_0 = EY_GIVEN_A0 * P_A0  # = 0.0
R_0 = L_0 + (1.0 - P_A0)  # = 0.6
L_1 = EY_GIVEN_A1 * P_A1  # = 0.24
R_1 = L_1 + (1.0 - P_A1)  # = 0.64

# True interventional values
DO_A0 = 0.4
DO_A1 = 0.5 - 1.25 * DELTA  # = 0.375


@pytest.fixture(scope="module")
def bounds() -> dict[int, BoundResult]:
    actions, rewards = _generate_observations(N_OBS, DELTA)
    return natural_bounds(actions, rewards, n_actions=2, n_bootstrap=50)


class TestArm0Bounds:
    def test_lower_bound_approx(self, bounds: dict[int, BoundResult]) -> None:
        assert abs(bounds[0].lower - L_0) < TOL

    def test_upper_bound_at_most_one(self, bounds: dict[int, BoundResult]) -> None:
        assert bounds[0].upper <= 1.0 + TOL

    def test_interventional_within_bounds_arm0(self, bounds: dict[int, BoundResult]) -> None:
        assert bounds[0].lower - TOL <= DO_A0 <= bounds[0].upper + TOL

    def test_p_a_approx(self, bounds: dict[int, BoundResult]) -> None:
        assert abs(bounds[0].p_a - P_A0) < TOL


class TestArm1Bounds:
    def test_lower_bound_approx(self, bounds: dict[int, BoundResult]) -> None:
        assert abs(bounds[1].lower - L_1) < TOL

    def test_upper_bound_approx(self, bounds: dict[int, BoundResult]) -> None:
        assert bounds[1].upper <= R_1 + TOL

    def test_interventional_within_bounds_arm1(self, bounds: dict[int, BoundResult]) -> None:
        assert bounds[1].lower - TOL <= DO_A1 <= bounds[1].upper + TOL

    def test_p_a_approx(self, bounds: dict[int, BoundResult]) -> None:
        assert abs(bounds[1].p_a - P_A1) < TOL


class TestBootstrapCIs:
    def test_ci_ordering_arm0(self, bounds: dict[int, BoundResult]) -> None:
        b = bounds[0]
        assert b.lower_ci_lo <= b.lower_ci_hi
        assert b.upper_ci_lo <= b.upper_ci_hi

    def test_ci_ordering_arm1(self, bounds: dict[int, BoundResult]) -> None:
        b = bounds[1]
        assert b.lower_ci_lo <= b.lower_ci_hi
        assert b.upper_ci_lo <= b.upper_ci_hi

    def test_ci_contains_point_estimate_arm1(self, bounds: dict[int, BoundResult]) -> None:
        b = bounds[1]
        assert b.lower_ci_lo <= b.lower <= b.lower_ci_hi
        assert b.upper_ci_lo <= b.upper <= b.upper_ci_hi
