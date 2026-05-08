"""v25.2 audit: measure the Z-conditional and U-conditional sample
ratios that the runner's ``reward_aligned`` behaviour policy actually
produces at ``alpha_conf=4`` on smooth and simpson reward profiles.

Read-only.  No envs, configs, or production code modified.

The runner constructs the behaviour policy via
``_make_behaviour_with_bias("reward_aligned", n_actions=8,
requires_latent=True)`` (``runner.py:164``), which dispatches via
``make_behaviour`` to ``RewardAlignedExplorer(n_actions=8,
requires_latent=True, bias_strength=...)``.  ``alpha_conf`` does NOT
reach this constructor — it's an env-side parameter that modulates
``observed_reward_prob`` (used for divergence diagnostics) but the
rollout's actual reward flow goes through ``do_reward(action)``
which depends only on ``_latent_state`` (Z), not U.  This audit
makes that structural fact empirically visible.

The runner passes ``latent=info["latent_U"]`` to ``select_action``
(``runner.py:840-841``); ``info["latent_U"]`` is a fresh
``Bernoulli(0.5)`` at every env reset, independent of Z
(``tabular_sepsis.py:266``).  So the policy peeks at U; U is
independent of Z; ⇒ buffer actions should be ~independent of Z, and
the Z-conditional sample ratio (the metric that determines whether
IPW vs naive can flip argmax under the simpson reward profile)
should sit near 1.0 regardless of profile.
"""
from __future__ import annotations

import numpy as np
import torch

from causal_rl.behaviour.reward_aligned import RewardAlignedExplorer
from causal_rl.envs.tabular_sepsis import (
    N_ACTIONS,
    OBS_STATES,
    TabularSepsisEnv,
)


def measure_confounding(
    profile: str, alpha_conf: float, n_samples: int, seed: int
) -> dict:
    """Sample ``n_samples`` (Z, U, action) tuples from the runner's
    actual ``RewardAlignedExplorer`` on a ``TabularSepsisEnv`` with
    the given profile and return per-Z and per-U per-arm sample
    counts.

    The env is built with ``n_envs=n_samples`` so a single ``reset``
    gives ``n_samples`` i.i.d. (Z, U) draws, matching the v25.1 gate
    fixture.
    """
    torch.manual_seed(seed)
    env = TabularSepsisEnv(
        cell=2,
        n_envs=n_samples,
        alpha_conf=alpha_conf,
        confounding_profile=profile,
        device="cpu",
    )
    obs, info = env.reset(seed=seed)

    # Match runner construction exactly (runner.py:164):
    #   self.behaviour_policy = self._make_behaviour_with_bias(
    #       config.behaviour, n_actions=n_actions, requires_latent=...)
    # which is RewardAlignedExplorer(n_actions=8, requires_latent=True,
    # bias_strength=config.bias_strength).
    policy = RewardAlignedExplorer(
        n_actions=N_ACTIONS,
        requires_latent=True,
        bias_strength=1.0,
    )

    z = (env._latent_state // OBS_STATES).long()
    u = info["latent_U"].view(-1).long()
    action, _logp = policy.select_action(obs, latent=info["latent_U"])
    a = action.view(-1).long()

    z_counts = np.zeros((2, N_ACTIONS), dtype=int)
    u_counts = np.zeros((2, N_ACTIONS), dtype=int)
    for zi, ui, ai in zip(z.tolist(), u.tolist(), a.tolist()):
        z_counts[zi, ai] += 1
        u_counts[ui, ai] += 1

    return {
        "profile": profile,
        "alpha_conf": alpha_conf,
        "n_samples": n_samples,
        "seed": seed,
        "z_counts": z_counts,
        "u_counts": u_counts,
    }


def report(result: dict) -> tuple[float, float]:
    z_counts = result["z_counts"]
    u_counts = result["u_counts"]
    profile = result["profile"]
    alpha = result["alpha_conf"]

    z_arm0_ratio = z_counts[0, 0] / max(z_counts[1, 0], 1)
    z_armN_ratio = z_counts[1, N_ACTIONS - 1] / max(z_counts[0, N_ACTIONS - 1], 1)
    u_arm0_ratio = u_counts[0, 0] / max(u_counts[1, 0], 1)
    u_armN_ratio = u_counts[1, N_ACTIONS - 1] / max(u_counts[0, N_ACTIONS - 1], 1)
    z_max = max(z_arm0_ratio, z_armN_ratio)
    u_max = max(u_arm0_ratio, u_armN_ratio)

    print(f"\n=== {profile} profile, alpha_conf={alpha} ===")
    print(f"Z-conditional counts (rows = Z, cols = arm):")
    print(f"  Z=0: {z_counts[0].tolist()}")
    print(f"  Z=1: {z_counts[1].tolist()}")
    print(f"  arm 0 Z=0/Z=1 ratio = {z_arm0_ratio:.2f}")
    print(f"  arm {N_ACTIONS-1} Z=1/Z=0 ratio = {z_armN_ratio:.2f}")
    print(f"  max Z side-arm ratio = {z_max:.2f}")
    print(f"U-conditional counts (rows = U, cols = arm):")
    print(f"  U=0: {u_counts[0].tolist()}")
    print(f"  U=1: {u_counts[1].tolist()}")
    print(f"  arm 0 U=0/U=1 ratio = {u_arm0_ratio:.2f}")
    print(f"  arm {N_ACTIONS-1} U=1/U=0 ratio = {u_armN_ratio:.2f}")
    print(f"  max U side-arm ratio = {u_max:.2f}")

    return z_max, u_max


def verdict(profile_z_max: float) -> str:
    if profile_z_max >= 5.0:
        return (
            f"VERDICT (simpson Z-ratio {profile_z_max:.2f} >= 5.0): "
            "Case A — strong Z-confounding.  The v25 sweep can proceed."
        )
    if profile_z_max >= 2.5:
        return (
            f"VERDICT (simpson Z-ratio {profile_z_max:.2f} in [2.5, 5.0)): "
            "Case B — weak Z-confounding.  The sweep will likely show "
            "muted IPW vs naive contrast.  Stop and decide."
        )
    return (
        f"VERDICT (simpson Z-ratio {profile_z_max:.2f} < 2.5): "
        "Case C — very weak Z-confounding.  ``alpha_conf=4`` is not "
        "generating the Z-conditional confounding the v25 sweep "
        "assumes.  v26 audit recommended before any further runs."
    )


def main() -> None:
    z_maxes: dict[str, float] = {}
    for profile in ("smooth", "simpson"):
        agg_z = np.zeros((2, N_ACTIONS), dtype=int)
        agg_u = np.zeros((2, N_ACTIONS), dtype=int)
        for seed in range(5):
            r = measure_confounding(
                profile, alpha_conf=4.0, n_samples=4000, seed=seed
            )
            agg_z += r["z_counts"]
            agg_u += r["u_counts"]
        z_max, u_max = report({
            "profile": profile,
            "alpha_conf": 4.0,
            "n_samples": 20000,
            "seed": "0-4 aggregated",
            "z_counts": agg_z,
            "u_counts": agg_u,
        })
        z_maxes[profile] = z_max

    print("\n=========================================")
    print(verdict(z_maxes["simpson"]))
    print("=========================================\n")


if __name__ == "__main__":
    main()
