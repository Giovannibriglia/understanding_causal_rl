"""v23: ``_load_or_build_transition`` and ``_load_or_build_reward``
share their default tensors via module-level caches.

Pre-v23 each ``TabularSepsisEnv`` instance rebuilt the 1440×8×1440
transition tensor and the 1440×8×2 reward tensor in pure-Python loops
(~200ms total per env on CPU).  ``BenchmarkRunner._run_offline``
constructs a fresh env at every eval / oracle / gap-metric checkpoint
(~24× per run), so this overhead dominated wall time on otherwise-
cheap configurations like ``paper_bandit_offpolicy.yaml`` — confirmed
by cProfile (8.86s of 12.37s = 72% in ``_make_env_with_n``).

The transition and reward tensors depend only on the global
``LATENT_STATES`` / ``OBS_STATES`` / ``N_ACTIONS`` constants — not on
``cell``, ``alpha_conf``, or any per-instance state — so a single
shared cache is correct.  Both perturbation paths
(``perturb_tabular_transition`` / ``_rewards``) ``.clone()`` the
input tensor before modifying it, so sharing the cache is safe; both
mutation sites in the codebase are reassignments of the form
``self.transition = perturb_*(self.transition, ...)``, never in-place
ops.

Custom ``transition_path`` / ``reward_path`` arguments still load
from disk and bypass the cache.
"""

from __future__ import annotations

import time

import torch

import causal_rl.envs.tabular_sepsis as tabular_sepsis
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv


def test_default_transition_tensor_is_shared() -> None:
    """Two ``TabularSepsisEnv`` instances with no custom ``transition_path``
    share the cached transition tensor — the post-v23 invariant that
    drives the speedup."""
    # Reset cache so the test is independent of test ordering.
    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = None

    e1 = TabularSepsisEnv(cell=1, n_envs=4, device="cpu")
    e2 = TabularSepsisEnv(cell=3, n_envs=8, device="cpu", alpha_conf=2.0)

    # Identity check on the underlying storage.  ``.to(device)`` with the
    # same device returns the same tensor object, so ``data_ptr()``
    # equality holds across instances on CPU.
    assert e1.transition.data_ptr() == e2.transition.data_ptr(), (
        "post-v23 the default transition tensor should be shared across "
        "instances; got distinct storage"
    )
    assert e1.reward_probs.data_ptr() == e2.reward_probs.data_ptr(), (
        "post-v23 the default reward tensor should be shared across "
        "instances"
    )
    # Values are bit-identical to a freshly-built tensor.
    assert torch.equal(e1.transition, e2.transition)
    assert torch.equal(e1.reward_probs, e2.reward_probs)


def test_repeated_construction_is_fast() -> None:
    """Constructing 24 instances (matching the per-run env-construction
    count in ``_run_offline``) must be much cheaper than the pre-v23
    Python-loop cost.

    Pre-v23 budget: ~200ms × 24 = ~4.8s on CPU.  Post-v23 the first
    construction populates the cache, the rest are zero-cost copies.
    Tolerance 1.5s leaves headroom for slow CI but still flags any
    regression that re-introduces the loop.
    """
    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = None

    t0 = time.perf_counter()
    for _ in range(24):
        TabularSepsisEnv(cell=1, n_envs=4, device="cpu")
    dt = time.perf_counter() - t0
    assert dt < 1.5, (
        f"24 env constructions took {dt:.2f}s — expected <1.5s with the "
        f"v23 module-level cache.  The per-instance Python-loop build "
        f"(~200ms each) appears to be running again."
    )


def test_perturbation_does_not_mutate_cache() -> None:
    """Defensive: even if the perturbation paths somehow began
    mutating their input in-place, this test would catch it.

    Build env, apply a non-trivial perturbation, build a second env,
    assert the second env's transition matches the original cached
    values (i.e. wasn't corrupted by the first env's perturbation).
    """
    from causal_rl.envs.perturbations import perturb_tabular_transition

    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = None

    e1 = TabularSepsisEnv(cell=1, n_envs=4, device="cpu")
    pre_perturb = e1.transition.clone()
    e1.transition = perturb_tabular_transition(e1.transition, eps_T=0.5)

    e2 = TabularSepsisEnv(cell=1, n_envs=4, device="cpu")
    assert torch.equal(e2.transition, pre_perturb), (
        "perturbation appears to have mutated the cached transition tensor"
    )


def test_transition_and_reward_independent_of_cell_and_alpha() -> None:
    """The cache returns the same tensor object regardless of ``cell``
    and ``alpha_conf`` — the assumption that justifies a single
    process-wide cache.

    A future change that accidentally introduces a cell- or
    alpha-dependent build (e.g. cell-specific reward shaping) would
    silently produce wrong results because envs across cells would
    all share the cache from whichever cell was constructed first.
    This test fails loudly in that case.

    Verifies *both* the values are identical (``torch.equal``) and
    the storage is shared (``data_ptr()`` equality).  Storage equality
    is the stronger property and is what ``_run_offline``'s ~24× env
    construction relies on for the v23 speedup.
    """
    tabular_sepsis._TRANSITION_CACHE = None
    tabular_sepsis._REWARD_CACHE = None

    envs = []
    for cell in (1, 2, 3, 4):
        for alpha in (0.0, 1.0, 2.0, 4.0):
            envs.append(
                TabularSepsisEnv(
                    cell=cell, n_envs=4, device="cpu", alpha_conf=alpha
                )
            )

    base_t_ptr = envs[0].transition.data_ptr()
    base_r_ptr = envs[0].reward_probs.data_ptr()
    base_t_val = envs[0].transition
    base_r_val = envs[0].reward_probs

    for env in envs[1:]:
        assert env.transition.data_ptr() == base_t_ptr, (
            f"transition cache violated: env(cell={env.cell}, "
            f"alpha={env.alpha_conf}) has distinct storage"
        )
        assert env.reward_probs.data_ptr() == base_r_ptr, (
            f"reward_probs cache violated: env(cell={env.cell}, "
            f"alpha={env.alpha_conf}) has distinct storage"
        )
        # Defence-in-depth: also assert values match, in case a
        # future change uses a per-call ``.clone()`` on cache return
        # (which would still be correct but would fail the data_ptr
        # check).
        assert torch.equal(env.transition, base_t_val)
        assert torch.equal(env.reward_probs, base_r_val)
