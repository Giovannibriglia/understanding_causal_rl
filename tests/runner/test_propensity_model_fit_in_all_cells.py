"""v13 Phase 3: propensity model is fit in every cell.

Pre-v13, the offline fit was gated on ``not self.cell_cfg.pi_b_known``
which skipped cells 1/3 (``pi_b_known=True``).  The visible consequence
in figures was that ``target_support_overlap`` was NaN in C1/C3 — only
C2 and C4 boxes appeared in ``causal_metrics_grid.pdf``'s third panel.

After dropping the predicate the propensity model is fit
unconditionally for discrete-action envs and ``target_support_overlap``
populates everywhere.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _smoke_cfg(out: Path, *, cell: int) -> RunnerConfig:
    return RunnerConfig(
        cell=cell,
        env_name="tabular-sepsis-v0",
        algorithm="dqn",
        behaviour="reward_aligned",
        seed=0,
        total_frames=200,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
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


def _last_eval_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "eval.csv").open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))[-1]


def _isfinite(val: object) -> bool:
    try:
        f = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


@pytest.mark.parametrize("cell", [1, 2, 3, 4])
def test_propensity_fitted_in_every_cell(tmp_path: Path, cell: int) -> None:
    """``target_support_overlap`` and ``propensity_calibration_ece`` are
    populated in all four cells.  Pre-v13 only C2 and C4 had values."""
    BenchmarkRunner(_smoke_cfg(tmp_path / f"c{cell}", cell=cell)).run()
    last = _last_eval_row(tmp_path / f"c{cell}")
    assert _isfinite(last["target_support_overlap"]), (
        f"target_support_overlap is NaN in cell {cell} — pre-v13 bug recurrence"
    )
    assert _isfinite(last["propensity_calibration_ece"]), (
        f"propensity_calibration_ece is NaN in cell {cell}"
    )
