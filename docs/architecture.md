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

## Throughput targets

- Tabular sepsis CPU: > 80k env-steps/s at `B=256`.
- Continuous ward CPU: > 30k env-steps/s at `B=256`.
- GPU targets are reported in `scripts/benchmark_speed.py` and documented as warnings in CI.
