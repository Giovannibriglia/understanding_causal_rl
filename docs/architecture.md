# Architecture

`causal-rl-benchmarks` is organised around typed registries, vectorised environments, and schema-stable CSV outputs.

## Design rationale

- Vectorised env API (`[B, ...]`) is the only first-class execution mode.
- Algorithms are split into model modules (`nets/`) and update logic (`algos/`).
- Behaviour policies are explicit and pluggable, including latent-dependent confounding variants.
- The runner logs fixed train/eval schemas across all components.
- Plotting reads CSV artifacts only, producing static paper-ready figures.

## Where to add new pieces

- New env: `src/causal_rl/envs/` + `envs/registry.py`.
- New algo: `src/causal_rl/algos/` + `algos/registry.py`.
- New behaviour: `src/causal_rl/behaviour/` + `behaviour/registry.py`.
- New divergence/metric hook: `src/causal_rl/metrics/`.
- New figure: `src/causal_rl/plotting/` + smoke test.

## Behaviour-independent algorithm deduplication

On-policy algorithms (PPO, A2C) collect their own experience and never use the
offline behaviour dataset, so the choice of behaviour policy in the matrix YAML
has no effect on their learned policy. Bandit algorithms (UCB, RCT) treat each
episode as i.i.d. and are similarly unaffected.

`build_runs` in `scripts/run_full_matrix.py` tracks a `_BEHAVIOUR_INDEPENDENT`
frozenset and, for these algorithms, includes only the **first** behaviour
listed in the config YAML. This eliminates redundant identical runs without
requiring any per-algorithm configuration: the deduplication is purely a
property of the learning algorithm's update rule.

If a new algorithm is behaviour-independent, add its registry key to
`_BEHAVIOUR_INDEPENDENT` in `run_full_matrix.py`.

## Checkpoint scheduling

`n_checkpoints_train` and `n_checkpoints_eval` are validated to be ≥ 2 and the
runner always logs at the first tick (`step=1`) and the last tick (`step=total`).
Intermediate checkpoints are distributed uniformly between them, clamped into
`[2, total-1]` so they never collide with the endpoints. See
`src/causal_rl/runner/scheduling.py::checkpoint_ticks` for the exact mapping.

## Throughput targets

- Tabular sepsis CPU: > 80k env-steps/s at `B=256`.
- Continuous ward CPU: > 30k env-steps/s at `B=256`.
- GPU targets are reported in `scripts/benchmark_speed.py` and documented as warnings in CI.
