# causal-rl-benchmarks
PyTorch-first vectorised benchmark suite for causal RL identifiability studies.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-008f68.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black.svg)](#)
[![codecov](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](#)

This repository is the experimental companion to *Debugging Reinforcement Learning through a Simple Causal Lens*. It benchmarks policies across four causal cells using controllable medical environments and logs the empirical identifiability gap \(\Delta_\varphi\), with TV as the primary operational metric.

## Quickstart
```bash
git clone <url> && cd understanding_causal_rl
```
```bash
pip install -e ".[dev]"
```
```bash
python scripts/run_single.py --cell 1 --env tabular-sepsis-v0 \
    --algorithm ppo --behaviour uniform --seed 0 \
    --total-frames 50000 --n-checkpoints-train 50 --n-checkpoints-eval 10 \
    --horizon 20 --rollout-horizon 20 --device cpu --output results/single_run
```
```bash
# Use the results directory printed by run_full_matrix (results/<config_name>_<timestamp>)
python scripts/make_all_figures.py --results results/<config_name>_<timestamp>
# Plots and tables will be written to outputs/<config_name>_<timestamp> by default.
```

## Reproducing the paper's headline figure
```bash
python scripts/run_full_matrix.py --seeds 0 1 2 --device cuda --config full_matrix
```

# After the run, use the printed results path (e.g. results/full_matrix_<timestamp>) as input to plotting:
```bash
python scripts/make_all_figures.py --results results/full_matrix_<timestamp>
```
# Plots and tables will be written to outputs/<config_name>_<timestamp> by default. Figures are separated by environment family and stored under 'tabular' and 'continuous' subfolders (e.g. outputs/full_matrix_<timestamp>/tabular).

Expected runtime: ~24h on a single A100 node, ~7 days on a single modern CPU box (tabular envs only).

## Launching Experiments

Single-run entrypoint: `scripts/run_single.py`.

```bash
python scripts/run_single.py \
  --cell 6 \
  --env continuous-ward-v0 \
  --algorithm ddpg \
  --behaviour reward_aligned \
  --seed 1 \
  --total-frames 100000 \
  --n-checkpoints-train 50 \
  --n-checkpoints-eval 10 \
  --horizon 50 \
  --rollout-horizon 50 \
  --device cuda \
  --output results/cell6_ddpg_seed1
```

Parameter semantics:

- `--cell`: causal cell id (`1..4`).
- `--env`: environment name (`tabular-sepsis-v0` or `continuous-ward-v0`).
- `--algorithm`: training algorithm from `algos/registry.py`.
- `--behaviour`: behaviour-policy family used for data collection.
- `--seed`: random seed for reproducible runs.
- `--total-frames`: total interaction/training frames. Must satisfy:
  `total_frames = n_episodes * episode_horizon_length`.
- `--n-checkpoints-train`: number of train rows to write in `train.csv`.
- `--n-checkpoints-eval`: number of eval rows to write in `eval.csv`.
- `--horizon`: explicit episode horizon passed to the environment.
- `--rollout-horizon`: rollout length used for online minibatch updates.
- `--device`: `cuda` or `cpu`.
- `--output`: output folder for `train.csv`, `eval.csv`, `meta.json`.

`eval.csv` also logs `eval_oracle_return_mean/std`, computed from an oracle policy (exact finite-horizon DP for tabular sepsis; model-based greedy oracle for continuous ward). Learning-curve eval panels overlay this oracle as a dashed black reference.

Matrix entrypoint: `scripts/run_full_matrix.py`. It reads frame/checkpoint controls and per-environment horizons from the YAML config:

- `total_frames`
- `n_checkpoints_train`
- `n_checkpoints_eval`
- `env_horizons` (per environment): specifies the episode horizon for each environment. The runner uses each environment's horizon as both the episode horizon and the rollout horizon. To override these defaults, `run_single.py` accepts explicit `--horizon` and `--rollout-horizon` flags.

## The four cells
| cell | shorthand | expose_Z | pi_b_known | typical id_status |
| --- | --- | --- | --- | --- |
| 1 | `mdp_known` | True | True | `id` |
| 2 | `mdp_unknown` | True | False | `id` / `partial_id` |
| 3 | `pomdp_known` | False | True | `partial_id` |
| 4 | `pomdp_unknown` | False | False | `partial_id` / `non_id` |

## Seven figures, six commands
- `identifiability_panel.pdf`
- `bandit_regret.pdf`
- `bound_width_panel.pdf`
- `bias_sweep.pdf`
- `sample_sweep.pdf`
- `headline_pure_divergence.pdf`
- `headline_fitted.pdf`

See `docs/figures.md` for the full mapping from figure to claim.

## Behaviour-policy families
`uniform` is the unbiased baseline with full support over actions.

Family 1 (`reward_aligned`, `reward_misaligned`) models treatment preference via value-like scores. `reward_misaligned` intentionally inverts/mis-specifies that signal and pushes toward harmful choices.

Family 2 (`information`, `certainty_seeking`) models uncertainty-driven data collection. `certainty_seeking` is the anti-information variant that over-selects predictable, low-coverage actions.

Family 3 (`curiosity`, `novelty_trap`) models novelty seeking. `novelty_trap` overweights risky/unstable novelty and tends to degrade downstream performance.

Full math and implementation notes: [docs/behaviour_policies.md](docs/behaviour_policies.md).

## The identifiability gap metric
\[
\Delta_\varphi(\pi)=\mathbb{E}_{(s,a)\sim d^\pi}\left[\varphi\!\left(P(R\mid a,s), P(R\mid do(a),s)\right)\right]
\]
with \(\varphi \in \{\mathrm{TV}, \mathrm{KL}, \chi^2, \sup\}\).  
TV is the primary instantiation because it is bounded, interpretable, and directly linked to return bias control.

Implementation detail: divergence values logged to `train.csv` and `eval.csv` are computed from deterministic policy rollouts at checkpoints (occupancy under the current learned policy), so they are comparable across algorithms within a cell.

## Architecture overview
The codebase is built around four registries (envs, algorithms, behaviour policies, divergences), a vectorised environment API, and a schema-fixed CSV runner. See [docs/architecture.md](docs/architecture.md) for extension points and throughput considerations.

## Repository layout
```text
causal-rl-benchmarks/
├── src/causal_rl/...
├── scripts/...
├── configs/...
├── tests/...
└── docs/...
```

## Adding a new component
### New environment
Implement `CausalEnv` in `src/causal_rl/envs/` and register in `envs/registry.py`.
### New algorithm
Implement `BaseAlgorithm` in `src/causal_rl/algos/` and register in `algos/registry.py`.
### New behaviour policy
Implement `BiasedExplorer` in `src/causal_rl/behaviour/` and register in `behaviour/registry.py`.
### New divergence
Add a pure function in `metrics/divergences.py` and wire it in `metrics/runner_hooks.py`.

## Citation
```bibtex
@misc{debugging_causal_rl_2026,
  title={Debugging Reinforcement Learning through a Simple Causal Lens},
  author={Anonymous},
  year={2026},
  note={Companion benchmarking repository: causal-rl-benchmarks}
}
```

## License
MIT. See [LICENSE](LICENSE).
