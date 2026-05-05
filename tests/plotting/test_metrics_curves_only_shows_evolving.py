"""v13 Phase 1 contract: ``metrics_curves`` plots only metrics that
genuinely evolve with the trained policy.

Pre-v13 the figure showed offline-buffer properties (min_propensity,
ess_ratio, ECE, ...) which are recomputed once at buffer collection
time and broadcast unchanged to every checkpoint, so the curves were
flat by construction.  v13 swapped them for the six Δ-divergence
variants — these *do* change with the trained policy's visit
distribution.

This test is the contract: if someone adds a buffer-property metric to
``DIAGNOSTIC_COLS`` it will fire.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from causal_rl.plotting.metrics_curves import DIAGNOSTIC_COLS
from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _smoke_cfg(out: Path, n_checkpoints_eval: int = 4) -> RunnerConfig:
    return RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="dqn",
        behaviour="reward_aligned",
        seed=0,
        total_frames=200,
        n_checkpoints_train=n_checkpoints_eval,
        n_checkpoints_eval=n_checkpoints_eval,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=400,
        offline_updates=40,
        alpha_conf=2.0,
        bias_strength=1.0,
        eval_perturbations=False,
        perturbation_grid_size=0,
    )


def _isfinite(val: str) -> bool:
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def test_metrics_curves_columns_evolve(tmp_path: Path) -> None:
    """Every column in ``metrics_curves.DIAGNOSTIC_COLS`` should have
    ≥ 2 distinct values across train.csv checkpoints; otherwise the
    curve is flat by construction and shouldn't be plotted as a time
    series."""
    BenchmarkRunner(_smoke_cfg(tmp_path / "run", n_checkpoints_eval=4)).run()
    with (tmp_path / "run" / "train.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2, "need at least 2 checkpoints to verify evolution"
    for col, _label in DIAGNOSTIC_COLS:
        values = {float(r[col]) for r in rows if _isfinite(r[col])}
        assert len(values) >= 2, (
            f"Column '{col}' in DIAGNOSTIC_COLS is flat (only saw {values}) — "
            "should be in static_diagnostics, not metrics_curves."
        )
