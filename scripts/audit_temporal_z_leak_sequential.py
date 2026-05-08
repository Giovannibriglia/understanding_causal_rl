"""v25.4 audit: measure how the Z-conditional sample ratio of the
sequential ``reward_aligned`` behaviour policy evolves across an
episode.  Read-only.

Hypothesis: even though ``RewardAlignedExplorer`` doesn't directly
peek at Z (v25.2/v25.3), the action it picks at step ``t`` is
influenced by ``obs_t``, and ``obs_t`` may accumulate Z-information
through transitions.  The v25.3 audit aggregated counts across all
timesteps and would average out any temporal accumulation.  v25.4
measures the per-timestep evolution.

Cell=2 (expose_z=True) vs cell=4 (expose_z=False) is a clean
contrast for the leak mechanism:
* Cell=2: Z is the 6th obs feature.  Mediated leak via ``obs.mean()``
  is structurally possible.
* Cell=4: Z is absent from obs.  Any temporal accumulation must come
  through the transition stochasticity producing different
  observable-state trajectories under different Z (which is how the
  env's confounding mechanism works in principle).

If both cells show flat ratios across timesteps, transitions don't
carry meaningful Z-signal into the buffer and v25.3's verdict
stands.  If late-episode ratios grow on cell=2 only, the leak is
obs-mediated.  If they grow on both, the leak is via transition
stochasticity.
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


def measure_temporal_confounding(
    cell: int,
    alpha_conf: float,
    n_episodes_per_seed: int,
    horizon: int,
    seed: int,
) -> dict:
    """Vectorised rollout across ``n_episodes_per_seed`` envs, recording
    per-step per-Z per-action counts.  Mirrors the runner's collector
    construction (``RewardAlignedExplorer(n_actions=8,
    requires_latent=True, bias_strength=1.0)`` + per-step
    ``select_action(obs, latent=info["latent_U"])``).
    """
    torch.manual_seed(seed)
    env = TabularSepsisEnv(
        cell=cell,
        n_envs=n_episodes_per_seed,
        horizon=horizon,
        alpha_conf=alpha_conf,
        device="cpu",
    )
    policy = RewardAlignedExplorer(
        n_actions=N_ACTIONS,
        requires_latent=True,
        bias_strength=1.0,
    )

    counts_by_step = np.zeros((horizon, 2, N_ACTIONS), dtype=int)
    obs, info = env.reset(seed=seed)
    for t in range(horizon):
        z = (env._latent_state // OBS_STATES).long()
        action, _ = policy.select_action(obs, latent=info["latent_U"])
        a = action.view(-1).long()
        for zi, ai in zip(z.tolist(), a.tolist()):
            counts_by_step[t, zi, ai] += 1
        obs, _r, _term, _trunc, info = env.step(action)
    return counts_by_step


def report(counts_by_step: np.ndarray, cell: int, alpha_conf: float) -> dict:
    horizon = counts_by_step.shape[0]
    print(f"\n=== sequential temporal Z-leak, cell={cell} "
          f"(expose_z={'True' if cell == 2 else 'False'}), "
          f"alpha_conf={alpha_conf} ===")

    print(f"\n{'step':>4} {'max_arm':>8} {'count_Z=0':>10} "
          f"{'count_Z=1':>10} {'ratio':>8}")
    print("-" * 46)
    per_step_max: list[float] = []
    for t in range(horizon):
        ratios: list[tuple[int, int, int, float]] = []
        for a in range(N_ACTIONS):
            c0 = int(counts_by_step[t, 0, a])
            c1 = int(counts_by_step[t, 1, a])
            if c0 + c1 < 20:
                continue
            r = max(c0 / max(c1, 1), c1 / max(c0, 1))
            ratios.append((a, c0, c1, r))
        if not ratios:
            print(f"{t:>4} {'--':>8} {'--':>10} {'--':>10} {'--':>8}")
            continue
        max_a, max_c0, max_c1, max_r = max(ratios, key=lambda x: x[3])
        per_step_max.append(max_r)
        print(f"{t:>4} {max_a:>8} {max_c0:>10} {max_c1:>10} {max_r:>8.2f}")

    early = per_step_max[: horizon // 2]
    late = per_step_max[horizon // 2 :]
    early_max = max(early) if early else 0.0
    late_max = max(late) if late else 0.0
    early_med = float(np.median(early)) if early else 0.0
    late_med = float(np.median(late)) if late else 0.0

    print(f"\nearly-episode (t < H/2): max={early_max:.2f}, median={early_med:.2f}")
    print(f"late-episode  (t >= H/2): max={late_max:.2f}, median={late_med:.2f}")
    growth = late_max - early_max
    print(f"max growth from early → late: {growth:+.2f}")
    print(f"==========================================")
    return {
        "cell": cell,
        "early_max": early_max,
        "early_med": early_med,
        "late_max": late_max,
        "late_med": late_med,
        "growth": growth,
    }


def main() -> None:
    horizon = 20
    n_episodes_per_seed = 200
    n_seeds = 5

    summaries: dict[int, dict] = {}
    for cell in (2, 4):
        agg = np.zeros((horizon, 2, N_ACTIONS), dtype=int)
        for seed in range(n_seeds):
            agg += measure_temporal_confounding(
                cell=cell, alpha_conf=4.0,
                n_episodes_per_seed=n_episodes_per_seed,
                horizon=horizon, seed=seed,
            )
        summaries[cell] = report(agg, cell=cell, alpha_conf=4.0)

    print("\n=========================================")
    print("VERDICT")
    print("=========================================")
    overall_late_max = max(s["late_max"] for s in summaries.values())
    cell2 = summaries[2]
    cell4 = summaries[4]
    print(f"cell=2 late-max ratio: {cell2['late_max']:.2f} "
          f"(growth {cell2['growth']:+.2f})")
    print(f"cell=4 late-max ratio: {cell4['late_max']:.2f} "
          f"(growth {cell4['growth']:+.2f})")
    print(f"overall late-max: {overall_late_max:.2f}")
    print()
    if overall_late_max >= 5.0:
        print("Significant late-episode temporal Z-leak.  Sequential")
        print("buffers may be partially confounded via transitions.")
        print("v26 sequential scope may shrink — investigate further.")
    elif overall_late_max >= 2.5:
        print("Weak temporal Z-leak.  Likely insufficient to support")
        print("claim-2/claim-1 evidence as currently interpreted, but")
        print("worth quantifying against the noise floor.")
    else:
        print("No meaningful temporal Z-leak.  v25.3 finding stands;")
        print("v26 sequential scope is full.")
    if cell2["late_max"] - cell4["late_max"] > 0.5:
        print()
        print("(cell=2 leaks more than cell=4 ⇒ obs-mediated component "
              "exists, but is small.)")
    print("=========================================\n")


if __name__ == "__main__":
    main()
