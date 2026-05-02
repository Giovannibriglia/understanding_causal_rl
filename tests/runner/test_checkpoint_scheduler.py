"""Acceptance tests for the checkpoint scheduler.

The scheduler must:
- always include both ``1`` and ``total`` in its output set,
- distribute intermediate ticks uniformly between them, clamped to
  ``[2, total-1]`` so they never collide with the endpoints,
- treat ``n_checkpoints < 2`` as ``2`` (clamped, not error).
"""

from __future__ import annotations

import csv
from pathlib import Path

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig
from causal_rl.runner.scheduling import checkpoint_ticks


def test_two_checkpoints_first_and_last() -> None:
    assert checkpoint_ticks(100, 2) == {1, 100}


def test_three_checkpoints_midpoint() -> None:
    assert checkpoint_ticks(100, 3) == {1, 50, 100}


def test_five_checkpoints_uniform_interior() -> None:
    assert checkpoint_ticks(100, 5) == {1, 25, 50, 75, 100}


def test_one_checkpoint_clamped_to_two() -> None:
    assert checkpoint_ticks(100, 1) == {1, 100}


def test_degenerate_total_smaller_than_n() -> None:
    # Intermediate ticks collide with the endpoints; the result is just {1, 2}.
    assert checkpoint_ticks(2, 5) == {1, 2}


def test_smoke_run_first_and_last_present(tmp_path: Path) -> None:
    """Smoke run with n_checkpoints_eval=2 must produce eval rows at step=1 and step=total."""
    out = tmp_path / "run"
    cfg = RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="cql",
        behaviour="uniform",
        seed=0,
        total_frames=40,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=80,
        offline_updates=40,
        eval_perturbations=False,
    )
    BenchmarkRunner(cfg).run()
    eval_path = out / "eval.csv"
    with eval_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f"expected 2 eval rows, got {len(rows)}"
    steps = [int(r["step"]) for r in rows]
    assert steps[0] == 1, f"first eval row at step={steps[0]}, expected 1"
    assert steps[-1] == cfg.total_frames, (
        f"last eval row at step={steps[-1]}, expected {cfg.total_frames}"
    )
