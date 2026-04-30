from __future__ import annotations

import csv
from pathlib import Path

from causal_rl.plotting.ablations import make_ablations
from causal_rl.plotting.cell_grid import make_cell_grid
from causal_rl.plotting.gap_curves import make_gap_curves
from causal_rl.plotting.headline_scatter import make_headline_scatter
from causal_rl.plotting.learning_curves import make_learning_curves
from causal_rl.plotting.summary_tables import make_summary_tables


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_plotting_smoke(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "figs"
    run_dir = results / "run0"
    train_cols = [
        "step",
        "cell",
        "env_name",
        "algorithm",
        "behaviour_policy",
        "seed",
        "train_return_mean",
        "delta_tv",
        "delta_kl",
        "delta_chi2",
        "delta_sup",
    ]
    eval_cols = [
        "step",
        "cell",
        "env_name",
        "algorithm",
        "behaviour_policy",
        "seed",
        "eval_return_mean",
        "eval_holdout_return_mean",
        "eval_oracle_return_mean",
        "delta_tv",
        "delta_kl",
        "delta_chi2",
        "delta_sup",
    ]
    train_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []
    for cell in range(1, 9):
        for step in [10, 20, 30]:
            train_rows.append(
                {
                    "step": step,
                    "cell": cell,
                    "env_name": "tabular-sepsis-v0",
                    "algorithm": "dqn",
                    "behaviour_policy": "uniform",
                    "seed": 0,
                    "train_return_mean": 0.1 * cell + step * 0.001,
                    "delta_tv": 0.05 * cell,
                    "delta_kl": 0.06 * cell,
                    "delta_chi2": 0.07 * cell,
                    "delta_sup": 0.08 * cell,
                }
            )
            eval_rows.append(
                {
                    "step": step,
                    "cell": cell,
                    "env_name": "tabular-sepsis-v0",
                    "algorithm": "dqn",
                    "behaviour_policy": "uniform",
                    "seed": 0,
                    "eval_return_mean": 1.0 - 0.05 * cell,
                    "eval_holdout_return_mean": 0.8 - 0.05 * cell,
                    "eval_oracle_return_mean": 1.2 - 0.03 * cell,
                    "delta_tv": 0.05 * cell,
                    "delta_kl": 0.06 * cell,
                    "delta_chi2": 0.07 * cell,
                    "delta_sup": 0.08 * cell,
                }
            )
    _write_csv(run_dir / "train.csv", train_cols, train_rows)
    _write_csv(run_dir / "eval.csv", eval_cols, eval_rows)

    make_learning_curves(results, out)
    make_gap_curves(results, out)
    make_headline_scatter(results, out)
    make_cell_grid(results, out)
    make_ablations(results, out)
    make_summary_tables(results, out)
    assert (out / "headline_gap_vs_generalisation.pdf").exists()
    assert (out / "cell_grid_8x4.pdf").exists()
    assert (out / "summary_results.tex").exists()
