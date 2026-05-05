"""v14: ``make_static_diagnostics`` renders two variants.

The bottom-right panel of ``static_diagnostics`` is parametrised across
``_max`` (``bound_width_max``) and ``_std`` (``bound_width_std``).  The
shared CSV scan loads data once; both figures are rendered.

This test is a focused contract for the v14 two-variant API: it builds
a tiny synthetic results tree (smaller than ``test_v11_diagnosis_figures``
fixture) and asserts both PDFs and PNGs land on disk.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from causal_rl.plotting.static_diagnostics import make_static_diagnostics  # noqa: E402

_HEADER = [
    "step",
    "cell",
    "env_name",
    "algorithm",
    "behaviour_policy",
    "seed",
    "min_propensity",
    "ess_ratio",
    "coverage_ess_histogram",
    "coverage_action_freq_min",
    "propensity_calibration_ece",
    "bound_width_max",
    "bound_width_std",
    "id_status",
]


def _write_run(
    run_dir: Path,
    *,
    cell: int,
    bound_width_max: float,
    bound_width_std: float,
    id_status: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "step": 100,
        "cell": cell,
        "env_name": "tabular-sepsis-v0",
        "algorithm": "dqn",
        "behaviour_policy": "uniform",
        "seed": 0,
        "min_propensity": 0.10,
        "ess_ratio": 0.95,
        "coverage_ess_histogram": 0.92,
        "coverage_action_freq_min": 0.10,
        "propensity_calibration_ece": 0.05,
        "bound_width_max": bound_width_max,
        "bound_width_std": bound_width_std,
        "id_status": id_status,
    }
    with (run_dir / "eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_HEADER)
        w.writeheader()
        w.writerow(row)


def test_renders_both_variants(tmp_path: Path) -> None:
    """v14 contract: ``static_diagnostics_max.{pdf,png}`` and
    ``static_diagnostics_std.{pdf,png}`` both land on disk."""
    results = tmp_path / "results"
    out = tmp_path / "fig"
    # Cells 1-4 with per-cell variation in the two informative aggregates.
    for cell in (1, 2, 3, 4):
        for seed in range(3):
            run_dir = results / f"run_c{cell}_s{seed}"
            _write_run(
                run_dir,
                cell=cell,
                bound_width_max=0.875 + 0.02 * cell + 0.005 * seed,
                bound_width_std=0.01 * cell + 0.002 * seed,
                id_status="id" if cell <= 2 else "partial_id",
            )
    make_static_diagnostics(results, out)
    assert (out / "static_diagnostics_max.pdf").exists()
    assert (out / "static_diagnostics_max.png").exists()
    assert (out / "static_diagnostics_std.pdf").exists()
    assert (out / "static_diagnostics_std.png").exists()


def test_skips_when_no_eval_csv(tmp_path: Path) -> None:
    """No ``eval.csv`` files → the function logs a skip and returns
    without crashing.  Neither PDF should appear."""
    results = tmp_path / "empty"
    results.mkdir(parents=True)
    out = tmp_path / "fig"
    make_static_diagnostics(results, out)
    assert not (out / "static_diagnostics_max.pdf").exists()
    assert not (out / "static_diagnostics_std.pdf").exists()
