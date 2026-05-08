"""v26 gate tests for ``RewardAlignedZExplorer``.

Three tests, in increasing coverage:

1. ``test_alpha_zero_is_uniform`` — backward-compat invariant.
   At ``α_conf=0``, the policy must produce a uniform distribution
   over actions.  Existing α=0 rows in summaries shouldn't change in
   expectation when configs migrate from ``reward_aligned`` to
   ``reward_aligned_z``.
2. ``test_alpha_four_confounds`` — the policy-isolated gate.
   At ``α_conf=4`` on a fresh rollout from a vectorised env, the
   Z-conditional sample ratio on the preferred arm must be ≥ 5:1.
   This is the analytical-prediction test; it doesn't go through the
   runner.
3. ``test_runner_integration_sequential`` — the end-to-end gate.
   A sequential rollout via the runner's actual collector path
   (``collect_offline_dataset`` with ``behaviour_policy=
   RewardAlignedZExplorer``) produces Z-conditional ratio ≥ 5:1.
   This is the test that catches the failure mode where the policy
   works in isolation but the runner constructs it incorrectly
   (wrong α_conf, missing ``latent_Z`` in info, wrong latent_source
   plumbing in the collector).

The brief's discipline: do not start any benchmark sweep until all
three tests pass green.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from causal_rl.behaviour.reward_aligned_z import RewardAlignedZExplorer
from causal_rl.envs.tabular_sepsis import (
    N_ACTIONS,
    OBS_STATES,
    TabularSepsisEnv,
)


def test_alpha_zero_is_uniform() -> None:
    """``α_conf=0`` ⇒ all weights = 1 ⇒ uniform.

    Sample 8000 actions across cell=2, count per arm.  Each arm
    should be within 3σ of the uniform expectation 1000.  σ on a
    Multinomial(n=8000, p=1/8) marginal is ≈ √(8000 × 1/8 × 7/8) ≈
    29.5; 3σ ≈ 89.
    """
    torch.manual_seed(0)
    n_samples = 8000
    env = TabularSepsisEnv(
        cell=2, n_envs=n_samples, alpha_conf=0.0, device="cpu",
    )
    obs, info = env.reset(seed=0)
    policy = RewardAlignedZExplorer(
        n_actions=N_ACTIONS, alpha_conf=0.0, bias_strength=1.0, env=env,
    )

    action, _ = policy.select_action(obs, latent=info["latent_Z"])
    counts = np.zeros(N_ACTIONS, dtype=int)
    for a in action.view(-1).long().tolist():
        counts[a] += 1

    expected = n_samples / N_ACTIONS
    sigma = math.sqrt(n_samples * (1 / N_ACTIONS) * (1 - 1 / N_ACTIONS))
    for a in range(N_ACTIONS):
        gap = abs(counts[a] - expected)
        assert gap < 3 * sigma, (
            f"arm {a}: count={counts[a]}, expected≈{expected:.0f}, "
            f"σ≈{sigma:.1f}, gap={gap:.1f} (> 3σ).  α=0 should give "
            f"uniform — backward-compat invariant violated."
        )


def test_alpha_four_confounds() -> None:
    """The policy-isolated gate: at ``α_conf=4`` on cell=2, the
    Z-conditional sample ratio on the preferred arm must be ≥ 5:1.

    Analytical: ``P(preferred|Z) = exp(4) / (exp(4) + 7) ≈ 0.886``;
    ``P(other|Z) ≈ 0.0163``.  Counts on n_samples=4000 vectorised:
    Z=0 has ~2000 samples, ~1772 land on the Z=0-preferred arm, ~32
    land on each non-preferred arm.  Z-ratio (Z=0 / Z=1) on the
    Z=0-preferred arm: ~1772/32 ≈ 55:1.
    """
    torch.manual_seed(0)
    n_samples = 4000
    env = TabularSepsisEnv(
        cell=2, n_envs=n_samples, alpha_conf=4.0, device="cpu",
    )
    obs, info = env.reset(seed=0)
    policy = RewardAlignedZExplorer(
        n_actions=N_ACTIONS, alpha_conf=4.0, bias_strength=1.0, env=env,
    )
    action, _ = policy.select_action(obs, latent=info["latent_Z"])

    z = (env._latent_state // OBS_STATES).long()
    a = action.view(-1).long()
    counts = np.zeros((2, N_ACTIONS), dtype=int)
    for zi, ai in zip(z.tolist(), a.tolist()):
        counts[zi, ai] += 1

    max_ratio = 0.0
    max_arm = -1
    for arm in range(N_ACTIONS):
        c0, c1 = int(counts[0, arm]), int(counts[1, arm])
        if c0 + c1 < 50:
            continue
        r = max(c0 / max(c1, 1), c1 / max(c0, 1))
        if r > max_ratio:
            max_ratio = r
            max_arm = arm

    assert max_ratio >= 5.0, (
        f"RewardAlignedZExplorer failed to confound at α_conf=4.\n"
        f"counts[Z=0]: {counts[0].tolist()}\n"
        f"counts[Z=1]: {counts[1].tolist()}\n"
        f"max Z-ratio {max_ratio:.2f} on arm {max_arm} (expected ≥ 5:1)."
    )


def test_runner_integration_sequential() -> None:
    """End-to-end gate: a buffer collected via the runner's actual
    ``collect_offline_dataset`` path with ``behaviour_policy=
    RewardAlignedZExplorer`` shows Z-conditional ratio ≥ 5:1.

    Catches failure modes the policy-isolated test can't:
    * runner constructing the policy without ``α_conf`` reaching it
      (the v25.2 bug for the legacy ``RewardAlignedExplorer``);
    * collector reading the wrong ``info[...]`` key (would raise or
      silently produce uniform actions);
    * ``latent_Z`` not exposed in the env's info dict.

    Mirrors v25.3 audit fixture: cell=2, horizon=20, 5 seeds × 80
    episodes (collected as a single 1600-env vectorised rollout per
    seed, totalling 8000 episodes × 20 steps = 160,000 transitions
    across the buffer).  Aggregates per-Z per-action counts and
    asserts max ratio ≥ 5:1.
    """
    from causal_rl.runner.collector import collect_offline_dataset

    n_envs = 80
    horizon = 20
    n_seeds = 5
    counts = np.zeros((2, N_ACTIONS), dtype=int)

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        env = TabularSepsisEnv(
            cell=2, n_envs=n_envs, horizon=horizon,
            alpha_conf=4.0, device="cpu",
        )
        policy = RewardAlignedZExplorer(
            n_actions=N_ACTIONS, alpha_conf=4.0, bias_strength=1.0, env=env,
        )
        # n_envs * horizon transitions per call
        buffer = collect_offline_dataset(
            env=env,
            behaviour_policy=policy,
            n_transitions=n_envs * horizon,
            expose_pi_b=True,
            expose_latent=True,
            seed=seed,
        )
        # The buffer logs ``latent`` (which is what the policy
        # received — i.e. latent_Z under v26's collector logic).
        # Use it directly: each transition's latent value IS the
        # Z value at that step.
        if buffer.latent is None:
            raise RuntimeError(
                "collector did not log latents — fix expose_latent path"
            )
        z = buffer.latent[: buffer.size].view(-1).long().clamp(0, 1)
        a = buffer.action[: buffer.size].view(-1).long()
        for zi, ai in zip(z.tolist(), a.tolist()):
            counts[zi, ai] += 1

    max_ratio = 0.0
    max_arm = -1
    for arm in range(N_ACTIONS):
        c0, c1 = int(counts[0, arm]), int(counts[1, arm])
        if c0 + c1 < 50:
            continue
        r = max(c0 / max(c1, 1), c1 / max(c0, 1))
        if r > max_ratio:
            max_ratio = r
            max_arm = arm

    assert max_ratio >= 5.0, (
        f"runner-integration gate failed: collector path didn't pass "
        f"latent_Z to RewardAlignedZExplorer or α_conf didn't reach the "
        f"policy.\ncounts[Z=0]: {counts[0].tolist()}\n"
        f"counts[Z=1]: {counts[1].tolist()}\n"
        f"max Z-ratio {max_ratio:.2f} on arm {max_arm} (expected ≥ 5:1)."
    )
