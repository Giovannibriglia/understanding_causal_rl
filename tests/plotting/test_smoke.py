from __future__ import annotations

import csv
import json
from pathlib import Path

from causal_rl.plotting.ablations import make_ablations
from causal_rl.plotting.bias_sweep import make_bias_sweep
from causal_rl.plotting.cell_grid import make_cell_grid
from causal_rl.plotting.gap_curves import make_gap_curves
from causal_rl.plotting.headline_scatter import make_headline_scatter
from causal_rl.plotting.identifiability_panel import make_identifiability_panel
from causal_rl.plotting.learning_curves import make_learning_curves
from causal_rl.plotting.sample_sweep import make_sample_sweep
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
        "id_status",
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
            id_status = ["id", "id", "partial_id", "non_id"][(cell - 1) % 4]
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
                    "id_status": id_status,
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
    make_identifiability_panel(results, out)
    # These return early when no data matches — just check they don't crash.
    make_bias_sweep(results, out)
    make_sample_sweep(results, out)
    assert (out / "headline_gap_vs_oracle.pdf").exists()
    assert (out / "cell_grid_8x4.pdf").exists()
    assert (out / "summary_results.tex").exists()
    assert (out / "identifiability_panel.pdf").exists()


def _write_bias_sweep_data(results_dir: Path) -> None:
    """Write synthetic eval.csv data with bias_strength for bias_sweep tests."""
    eval_cols = [
        "step",
        "cell",
        "env_name",
        "algorithm",
        "behaviour_policy",
        "seed",
        "eval_return_mean",
        "eval_oracle_return_mean",
        "id_status",
        "delta_tv",
        "delta_tv_ci_lo",
        "delta_tv_ci_hi",
        "min_propensity",
        "ess_ratio",
        "bias_strength",
    ]
    for cell in [1, 2, 3, 4]:
        for bs in [0.0, 0.5, 1.0]:
            run_dir = results_dir / f"cell{cell}_bs{int(bs*10)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            # Write meta.json
            (run_dir / "meta.json").write_text(json.dumps({"offline_transitions": 5000}))
            row = {
                "step": 100,
                "cell": cell,
                "env_name": "tabular-sepsis-v0",
                "algorithm": "dqn",
                "behaviour_policy": "uniform",
                "seed": 0,
                "eval_return_mean": 0.5 - 0.1 * bs,
                "eval_oracle_return_mean": 0.8,
                "id_status": ["id", "id", "partial_id", "non_id"][cell - 1],
                "delta_tv": 0.1 + 0.2 * bs,
                "delta_tv_ci_lo": 0.05 + 0.2 * bs,
                "delta_tv_ci_hi": 0.15 + 0.2 * bs,
                "min_propensity": 0.4 - 0.3 * bs,
                "ess_ratio": 0.9 - 0.7 * bs,
                "bias_strength": bs,
            }
            _write_csv(run_dir / "eval.csv", eval_cols, [row])


def test_bias_sweep_with_data(tmp_path: Path) -> None:
    """bias_sweep produces a PDF when proper data is present."""
    results = tmp_path / "bs_results"
    out = tmp_path / "bs_out"
    _write_bias_sweep_data(results)
    make_bias_sweep(results, out)
    assert (out / "bias_sweep.pdf").exists()


def _write_sample_sweep_data(results_dir: Path) -> None:
    """Write synthetic data with varying offline_transitions for sample_sweep."""
    eval_cols = [
        "step",
        "cell",
        "env_name",
        "algorithm",
        "behaviour_policy",
        "seed",
        "eval_return_mean",
        "eval_oracle_return_mean",
        "id_status",
        "bound_width_mean",
        "delta_tv",
    ]
    for cell in [1, 2, 3, 4]:
        for n_trans in [1000, 5000, 25000]:
            run_dir = results_dir / f"cell{cell}_n{n_trans}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "meta.json").write_text(json.dumps({"offline_transitions": n_trans}))
            row = {
                "step": 100,
                "cell": cell,
                "env_name": "tabular-sepsis-v0",
                "algorithm": "cql",
                "behaviour_policy": "reward_aligned",
                "seed": 0,
                "eval_return_mean": 0.3 + 0.1 * (n_trans / 25000),
                "eval_oracle_return_mean": 0.8,
                "id_status": ["id", "id", "partial_id", "non_id"][cell - 1],
                "bound_width_mean": 0.3,
                "delta_tv": 0.2,
            }
            _write_csv(run_dir / "eval.csv", eval_cols, [row])


def test_sample_sweep_with_data(tmp_path: Path) -> None:
    """sample_sweep produces a PDF when proper data is present."""
    results = tmp_path / "ss_results"
    out = tmp_path / "ss_out"
    _write_sample_sweep_data(results)
    make_sample_sweep(results, out)
    assert (out / "sample_sweep.pdf").exists()
