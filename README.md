# causal-rl-benchmarks
PyTorch-first vectorised benchmark suite for causal RL identifiability studies.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-008f68.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black.svg)](#)
[![codecov](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](#)

This repository is the experimental companion to *Debugging Reinforcement Learning through a Simple Causal Lens*. It benchmarks policies across four causal cells using controllable medical environments and logs the empirical identifiability gap $\Delta_\varphi$, with TV as the primary operational metric.

## Quickstart

```bash
git clone <url> && cd understanding_causal_rl
pip install -e ".[dev]"
```

**Sanity check (< 2 min on CPU):**

```bash
python scripts/run_full_matrix.py --config smoke_quick
```

**Single run:**

```bash
python scripts/run_single.py --cell 1 --env tabular-sepsis-v0 \
    --algorithm ppo --behaviour uniform --seed 0 \
    --total-frames 50000 --n-checkpoints-train 50 --n-checkpoints-eval 10 \
    --horizon 20 --rollout-horizon 20 --device cpu --output results/single_run
```

**Full paper run (~24 h on A100):**

```bash
python scripts/run_full_matrix.py --config paper --n-workers 8 --n-gpus 2
```

**Generate figures:**

```bash
# Use the results path printed by run_full_matrix
python scripts/make_all_figures.py --results results/<config_name>_<timestamp>
```

**SLURM array (HPC clusters):**

```bash
python scripts/run_full_matrix_slurm.py --config paper \
    --seeds 0 1 2 3 4 5 6 7 8 9 > commands.txt
sbatch --array=1-$(wc -l < commands.txt) slurm_array.sh
```

See [docs/experiment_plan.md](docs/experiment_plan.md) for the full Tier 0–4 experiment plan
and figure-claim mapping.

### Configurations

| File | Purpose |
|------|---------|
| `configs/smoke_quick.yaml` | < 2 min CPU import/shape sanity check |
| `configs/smoke.yaml` | Tabular smoke: cells 1–4 × dqn/cql/a2c × seeds [0,1] |
| `configs/smoke_v7.yaml` | v7 acceptance smoke (alpha_conf + bias_strength sweep; populates the `non_id` stratum) |
| `configs/smoke_v8.yaml` | v8 acceptance smoke (wide bias_strengths + perturbations on; generalisation regression mode) |
| `configs/horizon_sweep.yaml` | Horizon-impact study (Tier 1.5): 5/10/20/40/80 horizons |
| `configs/paper_smoke.yaml` | Paper-companion smoke; populates id/partial_id/non_id |
| `configs/paper_bandit.yaml` | Bandit family (UCB/UCB±/RCT) |
| `configs/paper.yaml` | Full headline run |
| `configs/bias_sweep.yaml` | Confounding sweep |
| `configs/sample_sweep.yaml` | Offline-data sweep |

## Launching Experiments

Single-run entrypoint: `scripts/run_single.py`.

```bash
python scripts/run_single.py \
  --cell 1 \
  --env tabular-sepsis-v0 \
  --algorithm dqn \
  --behaviour uniform \
  --seed 0 \
  --total-frames 200000 \
  --n-checkpoints-train 100 \
  --n-checkpoints-eval 50 \
  --horizon 20 \
  --rollout-horizon 20 \
  --alpha-conf 2.0 \
  --device cuda \
  --output results/cell1_dqn_seed0
```

Key `run_single.py` flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--cell` | required | Causal cell id (1–4) |
| `--env` | required | `tabular-sepsis-v0` or `continuous-ward-v0` |
| `--algorithm` | required | Algorithm name from `algos/registry.py` |
| `--behaviour` | required | Behaviour policy (uniform, reward_aligned, …) |
| `--seed` | required | Random seed |
| `--total-frames` | 50 000 | Training frames / offline updates |
| `--horizon` | None | Episode horizon (required for tabular) |
| `--alpha-conf` | 0.0 | Confounding strength α |
| `--n-eval-episodes` | 5 | Episodes per perturbation spec |
| `--eval-perturbations` / `--no-eval-perturbations` | on | Run TABULAR_DIAGONAL perturbed eval |
| `--output` | required | Output directory for CSVs |

`eval.csv` logs `eval_oracle_return_mean/std` (DP oracle for tabular; CEM for continuous),
`delta_tv` and `delta_tv_beh` (gap under learned vs. behaviour policy), and `ece`
(Expected Calibration Error). See [docs/metric.md](docs/metric.md) for definitions.

Matrix entrypoint: `scripts/run_full_matrix.py`:

```bash
python scripts/run_full_matrix.py --config paper --n-workers 4 --n-gpus 1
```

| YAML field | Description |
|------------|-------------|
| `total_frames` | Training frames |
| `n_checkpoints_train` / `n_checkpoints_eval` | CSV row count |
| `env_horizons` | Per-env episode horizon |
| `alpha_conf` | Single confounding level |
| `alpha_conf_sweep` | List of confounding levels (expands the factorial) |
| `eval_perturbations` | `true`/`false` — set `false` to halve sweep wall-time |
| `seeds` | List of random seeds |

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
$$
\Delta_\varphi(\pi)=\mathbb{E}_{(s,a)\sim d^\pi}\left[\varphi\!\left(P(R\mid a,s), P(R\mid do(a),s)\right)\right]
$$
with $\varphi \in \{\mathrm{TV}, \mathrm{KL}, \chi^2, \sup\}$.  
TV is the primary instantiation because it is bounded, interpretable, and directly linked to return bias control.

Implementation detail: divergence values logged to `train.csv` and `eval.csv` are computed from deterministic policy rollouts at checkpoints (occupancy under the current learned policy), so they are comparable across algorithms within a cell.

## Architecture overview
The codebase is built around four registries (envs, algorithms, behaviour policies, divergences), a vectorised environment API, and a schema-fixed CSV runner. See [docs/architecture.md](docs/architecture.md) for extension points and throughput considerations.

### Key docs

| Document | Contents |
|----------|----------|
| [docs/experiment_plan.md](docs/experiment_plan.md) | Tier 0–4 experiment plan, figure–claim mapping, quality gates |
| [docs/cells.md](docs/cells.md) | Causal cell definitions (expose_Z, expose_U, pi_b_known) |
| [docs/behaviour_policies.md](docs/behaviour_policies.md) | Behaviour policy families and their confounding patterns |
| [docs/metric.md](docs/metric.md) | Δ_TV, ECE, D_env_KS metric definitions |
| [docs/reproducing_paper.md](docs/reproducing_paper.md) | Step-by-step paper reproduction checklist |
| [docs/architecture.md](docs/architecture.md) | Codebase structure and extension guide |

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
