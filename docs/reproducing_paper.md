# Reproducing the paper

See [experiment_plan.md](experiment_plan.md) for the full Tier 0–4 plan with figure–claim mapping.

## Step 0 — Sanity check (< 2 min, CPU)

```bash
python scripts/run_full_matrix.py --config smoke_quick
```

All runs must exit 0 before proceeding.

## Step 1 — Full paper matrix (~24 h, A100)

```bash
# Local (8 workers, 2 GPUs)
python scripts/run_full_matrix.py --config paper --n-workers 10 --n-gpus 10 --device cuda

# SLURM array
python scripts/run_full_matrix_slurm.py --config paper \
    --seeds 0 1 2 3 4 5 6 7 8 9 > commands.txt
sbatch --array=1-$(wc -l < commands.txt) slurm_array.sh
```

The runner prints the results directory (e.g. `results/paper_<timestamp>`).

## Step 2 — Bias and sample sweeps

```bash
python scripts/run_full_matrix.py --config bias_sweep --n-workers 4 --n-gpus 10 --device cuda
python scripts/run_full_matrix.py --config sample_sweep --n-workers 4 --n-gpus 10 --device cuda
```

## Step 2b — Inspect summary (auto-generated)

`run_full_matrix.py` automatically writes `results/<run_id>/summary.json`
after every run (whether or not runs failed).  It contains per-cell / per-algo
aggregates, the headline regression R² values, and five pre-committed claim
verdicts.  To regenerate it manually:

```bash
python scripts/make_summary.py --results results/<run_id>
```

The `claims_evidence` block in `summary.json` shows whether the five pre-
committed claims are `supported`, `refuted`, `ambiguous`, or `not_yet_testable`.
Proceed to the paper run only once claim 1–3 read `supported` on `paper_smoke`.

## Step 3 — Build all figures

```bash
python scripts/make_all_figures.py --results results/paper_<timestamp>
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

- `cell`: causal cell (`1..4`).
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

- `outputs/{config_name}_{timestamp}/tabular/*.pdf`
- `outputs/{config_name}_{timestamp}/continuous/*.pdf`
- `outputs/{config_name}_{timestamp}/headline_gap_vs_oracle.pdf`
- `outputs/{config_name}_{timestamp}/cell_grid_8x4.pdf`
- `outputs/{config_name}_{timestamp}/ablations_*.pdf`

## Runtime expectations

- A100 x8 node: <= 24 hours (3 seeds full matrix).
- CPU-only quick mode: <= 20 minutes (`--quick`, single seed).
