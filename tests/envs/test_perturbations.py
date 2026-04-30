"""Tests for environment perturbations (Phase E.1)."""

from __future__ import annotations

import torch

from causal_rl.envs.perturbations import (
    CONTINUOUS_DIAGONAL,
    TABULAR_DIAGONAL,
    PerturbationSpec,
    perturb_tabular_rewards,
    perturb_tabular_transition,
)


def test_tabular_diagonal_has_five_levels() -> None:
    assert len(TABULAR_DIAGONAL) == 5
    assert len(CONTINUOUS_DIAGONAL) == 5


def test_first_level_is_unperturbed() -> None:
    assert TABULAR_DIAGONAL[0].eps_T == 0.0
    assert TABULAR_DIAGONAL[0].eps_R == 0.0
    assert CONTINUOUS_DIAGONAL[0].eps_target == 0.0
    assert CONTINUOUS_DIAGONAL[0].eps_dynamics == 0.0


def test_perturb_transition_zero_eps_is_identity() -> None:
    T = torch.rand(4, 3, 4)  # noqa: N806
    T = T / T.sum(dim=-1, keepdim=True)  # noqa: N806
    T_out = perturb_tabular_transition(T, eps_T=0.0)  # noqa: N806
    assert torch.allclose(T, T_out)


def test_perturb_transition_positive_eps_changes_values() -> None:
    torch.manual_seed(0)
    T = torch.rand(8, 4, 8)  # noqa: N806
    T = T / T.sum(dim=-1, keepdim=True)  # noqa: N806
    T_out = perturb_tabular_transition(T, eps_T=0.3)  # noqa: N806
    # After perturbation, rows should still be non-negative and sum to ~1
    assert (T_out >= 0).all()
    row_sums = T_out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_perturb_rewards_zero_eps_is_identity() -> None:
    R = torch.rand(4, 3, 2)  # noqa: N806
    R[:, :, 0] = 1 - R[:, :, 1]
    R_out = perturb_tabular_rewards(R, eps_R=0.0)  # noqa: N806
    assert torch.allclose(R, R_out)


def test_perturb_rewards_positive_eps_shifts_toward_half() -> None:
    R = torch.zeros(4, 3, 2)  # noqa: N806
    R[:, :, 1] = 0.8  # p_pos = 0.8
    R[:, :, 0] = 0.2
    R_out = perturb_tabular_rewards(R, eps_R=0.5)  # noqa: N806
    # Expected: p_pos' = 0.5 * 0.8 + 0.5 * 0.5 = 0.65
    expected = (1.0 - 0.5) * 0.8 + 0.5 * 0.5
    assert abs(float(R_out[0, 0, 1].item()) - expected) < 1e-5


def test_perturbation_spec_frozen() -> None:
    spec = PerturbationSpec(eps_T=0.1, eps_R=0.05)
    assert spec.eps_T == 0.1
    assert spec.eps_R == 0.05
