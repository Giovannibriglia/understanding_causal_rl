"""v25: ``TabularSepsisEnv`` accepts a ``confounding_profile`` kwarg
that selects between the legacy ``"smooth"`` reward profile and a new
``"simpson"`` profile producing Simpson's-paradox confounding.

These tests guard the contract:

* ``smooth`` produces *byte-identical* reward and transition tensors to
  pre-v25 (regression guard for paper.yaml / smoke_v8 / coverage_stress
  / paper_bandit summary.json reproducibility).
* ``simpson`` produces a *different* reward tensor (sanity that the
  dispatch is wired).

The argmax-flip behavioural gate that motivates v25 lives in
``tests/algos/test_simpson_argmax_flip.py``.
"""

from __future__ import annotations

import pytest
import torch

import causal_rl.envs.tabular_sepsis as tabular_sepsis
from causal_rl.envs.tabular_sepsis import OBS_STATES, TabularSepsisEnv


def _reset_caches() -> None:
    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = {}


def test_smooth_profile_byte_identical() -> None:
    """The default ``confounding_profile`` is ``"smooth"`` and must
    produce reward + transition tensors equal to those built with no
    profile argument at all.  This is the regression guard for every
    config that doesn't set ``confounding_profile`` — paper.yaml,
    smoke_v8.yaml, coverage_stress.yaml, paper_bandit.yaml all
    continue to produce byte-identical summary.json hashes.

    Also asserts equality across ``cell`` so the v23/v24 cache
    invariant (cell-independence under ``smooth``) still holds.
    """
    _reset_caches()
    e_default = TabularSepsisEnv(cell=1, n_envs=4, device="cpu")
    e_explicit = TabularSepsisEnv(
        cell=1, n_envs=4, device="cpu", confounding_profile="smooth"
    )
    e_other_cell = TabularSepsisEnv(
        cell=3, n_envs=4, device="cpu", confounding_profile="smooth"
    )
    assert torch.equal(e_default.reward_probs, e_explicit.reward_probs), (
        "explicit confounding_profile='smooth' should produce reward "
        "tensors bit-identical to the default (regression for paper.yaml)"
    )
    assert torch.equal(e_default.transition, e_explicit.transition)
    assert torch.equal(e_default.reward_probs, e_other_cell.reward_probs), (
        "smooth profile should be cell-independent (cache invariant)"
    )
    assert torch.equal(e_default.transition, e_other_cell.transition)


def test_simpson_profile_differs_from_smooth() -> None:
    """Smoke test for dispatch: ``simpson`` produces a different
    reward tensor than ``smooth``.  Specifically, under Z=0 arm 0
    has p_pos=0.85 (P_HIGH) and arm N-1 has p_pos=0.10 (P_LOW); under
    Z=1 the side arms swap.  Middle arms have p_pos=0.55 (P_MID)
    under both Z values.
    """
    _reset_caches()
    e_smooth = TabularSepsisEnv(
        cell=1, n_envs=4, device="cpu", confounding_profile="smooth"
    )
    e_simpson = TabularSepsisEnv(
        cell=1, n_envs=4, device="cpu", confounding_profile="simpson"
    )
    assert not torch.equal(e_simpson.reward_probs, e_smooth.reward_probs), (
        "simpson dispatch should produce a distinct reward tensor"
    )
    # Transitions are profile-independent — verify they're still shared.
    assert torch.equal(e_simpson.transition, e_smooth.transition)

    # Spot-check the Simpson layout at z=0.
    p_pos_z0_a0 = float(e_simpson.reward_probs[0, 0, 1].item())
    p_pos_z0_a7 = float(e_simpson.reward_probs[0, 7, 1].item())
    p_pos_z0_a3 = float(e_simpson.reward_probs[0, 3, 1].item())
    assert abs(p_pos_z0_a0 - 0.85) < 1e-6, p_pos_z0_a0
    assert abs(p_pos_z0_a7 - 0.10) < 1e-6, p_pos_z0_a7
    assert abs(p_pos_z0_a3 - 0.55) < 1e-6, p_pos_z0_a3
    # And the Z=1 swap.
    p_pos_z1_a0 = float(e_simpson.reward_probs[OBS_STATES, 0, 1].item())
    p_pos_z1_a7 = float(e_simpson.reward_probs[OBS_STATES, 7, 1].item())
    assert abs(p_pos_z1_a0 - 0.10) < 1e-6, p_pos_z1_a0
    assert abs(p_pos_z1_a7 - 0.85) < 1e-6, p_pos_z1_a7


def test_invalid_profile_raises() -> None:
    with pytest.raises(ValueError, match="confounding_profile"):
        TabularSepsisEnv(
            cell=1, n_envs=4, device="cpu", confounding_profile="invalid"
        )
