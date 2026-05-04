"""Cell-grid panel must contain only its own cell's rows.

If a logger ever emits the wrong ``cell`` value on intermediate checkpoint
rows, ``make_cell_grid`` will assert here rather than silently produce a
mislabelled figure.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from causal_rl.plotting.cell_grid import make_cell_grid  # noqa: E402

_HEADER = [
    "step",
    "cell",
    "env_name",
    "algorithm",
    "behaviour_policy",
    "seed",
    "delta_tv",
    "eval_return_mean",
    "eval_oracle_return_mean",
]


def _row(cell: int, algo: str, beh: str, seed: int, dtv: float, ret: float) -> dict[str, object]:
    return {
        "step": 100,
        "cell": cell,
        "env_name": "tabular-sepsis-v0",
        "algorithm": algo,
        "behaviour_policy": beh,
        "seed": seed,
        "delta_tv": dtv,
        "eval_return_mean": ret,
        "eval_oracle_return_mean": ret + 2.0,
    }


def _write_run(run_dir: Path, rows: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "eval.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADER)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_cell_grid_partitions_correctly(tmp_path: Path) -> None:
    results = tmp_path / "res"
    out = tmp_path / "fig"
    for cell in (1, 2, 3, 4):
        _write_run(
            results / f"run_cell{cell}",
            [_row(cell, "dqn", "uniform", 0, 0.05 * cell, 10.0 - cell)],
        )
    make_cell_grid(results, out)
    assert (out / "cell_grid_1x4.png").exists()


def test_cell_grid_asserts_on_bleed(tmp_path: Path) -> None:
    """A run dir whose CSV mixes cell IDs must trigger the assert."""
    results = tmp_path / "res"
    out = tmp_path / "fig"
    _write_run(
        results / "run_bleed",
        [
            _row(1, "dqn", "uniform", 0, 0.04, 9.0),
            _row(3, "dqn", "uniform", 0, 0.20, 6.0),  # mislabelled bleed
        ],
    )
    with pytest.raises(AssertionError, match="contains rows from"):
        make_cell_grid(results, out)
