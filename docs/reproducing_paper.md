# Reproducing the paper

## Full matrix

```bash
python scripts/run_full_matrix.py \
  --seeds 0 1 2 \
  --device cuda \
  --config full_matrix
```

# The runner prints the selected results directory (results/full_matrix_<timestamp>); use that path as input to plotting.

## Build all figures

```bash
# Use the results path printed by run_full_matrix (e.g. results/full_matrix_<timestamp>)
python scripts/make_all_figures.py --results results/full_matrix_<timestamp>
# Output will be written to outputs/full_matrix_<timestamp> by default.
# Figures are separated by environment family and stored under 'tabular' and 'continuous' subfolders.
```

## Single run (debug/ablation)

```bash
python scripts/run_single.py \
  --cell 3 \
  --env tabular-sepsis-v0 \
  --algorithm cql \
  --behaviour uniform \
  --seed 0 \
  --total-frames 100000 \
  --n-checkpoints-train 50 \
  --n-checkpoints-eval 10 \
  --horizon 20 \
  --rollout-horizon 20 \
  --device cuda \
  --output results/single_debug
```

## Experiment parameters

`run_single.py` parameters:

- `cell`: causal cell (`1..8`).
- `env`: registered env id.
- `algorithm`: registered algorithm id.
- `behaviour`: registered behaviour-policy id.
- `seed`: run seed.
- `total_frames`: total training/data-collection frames.
- `n_checkpoints_train`: number of train checkpoints logged.
- `n_checkpoints_eval`: number of eval checkpoints logged.
- `horizon`: environment episode horizon.
- `rollout_horizon`: rollout length for online minibatch updates.

Constraint:

- `total_frames` must be divisible by `horizon`.
- Equivalently: `total_frames = n_episodes * horizon`.

`run_full_matrix.py` reads the same semantics from YAML:

- `total_frames`
- `n_checkpoints_train`
- `n_checkpoints_eval`
- `env_horizons` (per environment): specifies the episode horizon for each environment. The runner uses each environment's horizon as both the episode horizon and the rollout horizon. To override these defaults, `run_single.py` accepts explicit `--horizon` and `--rollout-horizon` flags.

## Expected outputs

- `outputs/{config_name}_{timestamp}/learning_curves_cell{1..8}.pdf`
- `outputs/{config_name}_{timestamp}/gap_curves_cell{1..8}.pdf`
- `outputs/{config_name}_{timestamp}/headline_gap_vs_generalisation.pdf`
- `outputs/{config_name}_{timestamp}/cell_grid_8x4.pdf`
- `outputs/{config_name}_{timestamp}/ablations_*.pdf`

## Runtime expectations

- A100 x8 node: <= 24 hours (3 seeds full matrix).
- CPU-only quick mode: <= 20 minutes (`--quick`, single seed).
