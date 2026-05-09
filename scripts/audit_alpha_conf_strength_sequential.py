"""v25.3 audit: measure the Z-conditional sample ratio that the runner's
sequential ``reward_aligned`` behaviour policy actually produces at
``alpha_conf=4`` on the sequential ``TabularSepsisEnv`` (horizon=20,
matching paper.yaml / smoke_v8 / coverage_stress).

Read-only.  No envs, configs, or production code modified.

Phase 0 findings (traced before writing this script):

* The runner constructs the same ``RewardAlignedExplorer`` for both
  bandit and sequential paths (``runner.py:164``).  No sequential-
  specific class.
* The off-policy collector (``collector.py:99-102``) calls
  ``policy.select_action(obs, latent=info["latent_U"])`` once per
  step.  Identical to the bandit construction in v25.2.
* ``alpha_conf`` does NOT reach ``RewardAlignedExplorer.__init__`` on
  this path either (same gap as bandits).
* The one structural difference: on cell=2 (expose_z=True), the obs
  includes Z as the 6th feature (``tabular_sepsis.py:240``), so
  ``state_score = obs.mean(dim=-1)`` picks up a 1/6 contribution
  from Z.  This is the only pathway for Z to influence the policy's
  actions in the runner's actual rollout — and it's an indirect
  pathway, not an explicit Z-peek.  On cell=4 (expose_z=False) Z
  doesn't appear in obs and the policy is fully Z-blind.

Audit measures both cell=2 (where indirect Z-peeking is possible)
and cell=4 (where Z-peeking is structurally impossible) at
alpha_conf=4, horizon=20, aggregated over 5 seeds × 80 episodes.
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


def measure_sequential_confounding(
    cell: int,
    alpha_conf: float,
    n_episodes: int,
    horizon: int,
    seed: int,
) -> dict:
    """Roll out ``n_episodes`` with the runner's actual
    ``RewardAlignedExplorer`` on a sequential ``TabularSepsisEnv``
    and return per-Z per-action counts aggregated across all
    timesteps and per-timestep.

    Mirrors ``collector.collect_offline_dataset`` in policy
    construction and the per-step ``select_action(obs,
    latent=info["latent_U"])`` call.
    """
    torch.manual_seed(seed)
    env = TabularSepsisEnv(
        cell=cell,
        n_envs=n_episodes,  # vectorised across episodes
        horizon=horizon,
        alpha_conf=alpha_conf,
        device="cpu",
    )
    policy = RewardAlignedExplorer(
        n_actions=N_ACTIONS,
        requires_latent=True,
        bias_strength=1.0,
    )

    counts = np.zeros((2, N_ACTIONS), dtype=int)
    counts_by_step = np.zeros((horizon, 2, N_ACTIONS), dtype=int)

    obs, info = env.reset(seed=seed)
    for t in range(horizon):
        z = (env._latent_state // OBS_STATES).long()
        action, _logprob = policy.select_action(obs, latent=info["latent_U"])
        a = action.view(-1).long()
        for zi, ai in zip(z.tolist(), a.tolist()):
            counts[zi, ai] += 1
            counts_by_step[t, zi, ai] += 1
        obs, _reward, _term, _trunc, info = env.step(action)

    return {
        "cell": cell,
        "alpha_conf": alpha_conf,
        "n_episodes": n_episodes,
        "horizon": horizon,
        "seed": seed,
        "counts": counts,
        "counts_by_step": counts_by_step,
    }


def report(result: dict, label: str) -> float:
    counts = result["counts"]
    by_step = result["counts_by_step"]
    horizon = result["horizon"]

    print(f"\n=== sequential reward_aligned, {label} ===")
    print(f"alpha_conf={result['alpha_conf']}, "
          f"n_episodes={result['n_episodes']}, "
          f"horizon={horizon}")

    print(f"\nAggregated (all timesteps):")
    print(f"counts[Z=0, :] = {counts[0].tolist()}")
    print(f"counts[Z=1, :] = {counts[1].tolist()}")

    ratios: list[tuple[int, float | None]] = []
    for a in range(N_ACTIONS):
        c0 = int(counts[0, a])
        c1 = int(counts[1, a])
        if c0 + c1 < 50:
            ratios.append((a, None))
            continue
        r = max(c0 / max(c1, 1), c1 / max(c0, 1))
        ratios.append((a, r))

    valid = [(a, r) for a, r in ratios if r is not None]
    max_r: float | None = None
    if valid:
        max_a, max_r = max(valid, key=lambda x: x[1] if x[1] is not None else 0.0)
        print(f"per-action Z-ratios: " +
              ", ".join(f"a{a}={r:.2f}" for a, r in valid))
        print(f"max per-action Z-ratio: arm {max_a} at {max_r:.2f}:1")
    else:
        print(f"all arms too sparse to compute ratios")

    if valid and max_r is not None:
        max_a = max(valid, key=lambda x: x[1] if x[1] is not None else 0.0)[0]
        print(f"\nPer-timestep ratio for arm {max_a}:")
        for t in range(horizon):
            c0 = int(by_step[t, 0, max_a])
            c1 = int(by_step[t, 1, max_a])
            if c0 + c1 < 10:
                continue
            r = max(c0 / max(c1, 1), c1 / max(c0, 1))
            print(f"  t={t:2d}: counts=({c0}, {c1}), ratio={r:.2f}")

    print("")
    if max_r is not None and max_r >= 5.0:
        print(f"  [{label}] >= 5:1 — strong Z-confounding on this cell")
    elif max_r is not None and max_r >= 2.5:
        print(f"  [{label}] in [2.5, 5.0) — weak Z-confounding")
    else:
        print(f"  [{label}] < 2.5 — no Z-confounding")
    print(f"==========================================")
    return max_r if max_r is not None else 0.0


def main() -> None:
    horizon = 20
    n_episodes_per_seed = 80
    n_seeds = 5
    n_episodes = n_episodes_per_seed * n_seeds

    cell_max_ratios: dict[int, float] = {}
    for cell in (2, 4):
        agg_counts = np.zeros((2, N_ACTIONS), dtype=int)
        agg_by_step = np.zeros((horizon, 2, N_ACTIONS), dtype=int)
        for seed in range(n_seeds):
            r = measure_sequential_confounding(
                cell=cell, alpha_conf=4.0,
                n_episodes=n_episodes_per_seed, horizon=horizon,
                seed=seed,
            )
            agg_counts += r["counts"]
            agg_by_step += r["counts_by_step"]
        max_r = report({
            "cell": cell, "alpha_conf": 4.0,
            "n_episodes": n_episodes, "horizon": horizon,
            "seed": "0-4 aggregated",
            "counts": agg_counts, "counts_by_step": agg_by_step,
        }, label=f"cell={cell} (expose_z={'True' if cell == 2 else 'False'})")
        cell_max_ratios[cell] = max_r

    headline = max(cell_max_ratios.values())
    print(f"\n=========================================")
    print(f"VERDICT: max Z-ratio across cells = {headline:.2f}")
    if headline >= 5.0:
        print(f"  Case A — sequential is intact.")
        print(f"  Bug is bandit-specific; v26 fix scope is small.")
    elif headline >= 2.5:
        print(f"  Case B — weak sequential confounding.")
        print(f"  Sequential evidence may be partially affected.")
        print(f"  Stop and decide before drafting v26.")
    else:
        print(f"  Case C — sequential has the same bug as bandits.")
        print(f"  v26 fix scope must include sequential.")
        print(f"  Paper timeline implications.  Stop hard.")
    print(f"=========================================\n")


if __name__ == "__main__":
    main()
