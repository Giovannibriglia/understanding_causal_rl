"""Estimate wall-time for a matrix config.

Heuristic per-run time based on knobs that materially affect cost
(eval_perturbations, perturbation_grid_size, n_eval_episodes, total_frames,
n_envs, offline_updates).  Numbers are calibrated against the v9 / v10
profile of ``smoke_v8`` on an 8-core CPU box.

Usage::

    python scripts/estimate_run_time.py configs/smoke_v8.yaml --n-workers 4
"""

from __future__ import annotations

import argparse
import math
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from causal_rl.config.schemas import MatrixConfig, load_yaml  # noqa: E402

# Calibration baselines.  These are the per-run numbers measured on an
# 8-core CPU box with the v10 perturbation eval (single end-of-training
# call, vectorised bisim, env built once per spec):
_BASE_RUN_S = 51.0           # base run with --no-eval-perturbations
_PERT_PER_SPEC_S = 0.9       # rollout (3 ep × n_envs) + bisim + KS, single call
_BASE_OVERHEAD_S = 1.5       # env construction + RNG save/restore


def _axis_count(values: object) -> int:
    if values is None:
        return 1
    if isinstance(values, (list, tuple)):
        return max(1, len(values))
    return 1


def estimate_total_runs(matrix: MatrixConfig) -> int:
    cells = len(matrix.cells)
    envs = len(matrix.envs)
    algos = len(matrix.algorithms)
    behaviours = len(matrix.behaviours)
    seeds = len(matrix.seeds) if matrix.seeds else 1
    alpha = _axis_count(matrix.alpha_conf_sweep)
    horizon = _axis_count(matrix.horizon_sweep)
    # When ``coverage_regimes`` is set it replaces the bias×offline axes
    # for the regime dimension (each regime is one entry).  Otherwise the
    # bias_strengths / offline_transitions_sweep axes apply.
    if matrix.coverage_regimes:
        bias = 1
        offline_t = 1
        regimes = len(matrix.coverage_regimes)
    else:
        bias = _axis_count(matrix.bias_strengths)
        offline_t = _axis_count(matrix.offline_transitions_sweep)
        regimes = 1
    return (
        cells
        * envs
        * algos
        * behaviours
        * seeds
        * alpha
        * horizon
        * bias
        * offline_t
        * regimes
    )


def estimate_per_run_seconds(matrix: MatrixConfig) -> float:
    base = _BASE_RUN_S
    if not matrix.eval_perturbations:
        return base
    grid = matrix.perturbation_grid_size or 5
    if grid <= 0:
        grid = 5
    grid = min(grid, 5)
    # Linear in grid size for the per-spec work; one constant overhead.
    rollout_factor = max(0.5, matrix.n_eval_episodes / 3.0)
    return base + _BASE_OVERHEAD_S + grid * _PERT_PER_SPEC_S * rollout_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a matrix YAML.")
    parser.add_argument(
        "--n-workers", type=int, default=1, help="Parallel workers."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        candidate = ROOT / "configs" / args.config.name
        if candidate.is_file():
            args.config = candidate
    data = load_yaml(args.config)
    matrix = MatrixConfig.model_validate(data)

    n_runs = estimate_total_runs(matrix)
    per_run = estimate_per_run_seconds(matrix)
    sequential = n_runs * per_run
    parallel = sequential / max(1, args.n_workers)

    print(f"Config:       {args.config}")
    print(f"Runs:         {n_runs}")
    print(f"Per-run:      {per_run:.1f} s")
    print(f"Sequential:   {_fmt_time(sequential)}")
    print(f"At {args.n_workers}x:        {_fmt_time(parallel)}")
    if matrix.eval_perturbations:
        grid = matrix.perturbation_grid_size or 5
        print(
            f"  (perturbations on, grid={grid}, n_eval_episodes={matrix.n_eval_episodes})"
        )
    else:
        print("  (perturbations off)")


def _fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    return f"{hours:.1f} h"


if __name__ == "__main__":
    main()
