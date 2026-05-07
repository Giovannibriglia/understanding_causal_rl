"""v20: ``make_bandit_regret`` plots within-run cumulative regret per
cell with a clean two-facet layout (C1, C3).

Pre-v20 the figure had four bugs (identical Δ panels, across-run
cumsum, broken regret comparisons, fictional Δ axis).  These tests
guard the post-v20 contract: per-run cumsum, monotone within a run,
graceful skip when there's nothing to plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_rl.plotting.bandit_regret import (
    _RUN_GROUPING,
    _within_run_cum_regret,
    make_bandit_regret,
)


def _write_eval_csv(
    path: Path,
    *,
    env_name: str,
    cell: int,
    algorithm: str,
    seed: int,
    n_rows: int = 5,
    base_eval: float = 0.5,
    base_oracle: float = 0.9,
    behaviour_policy: str = "uniform",
    alpha_conf: float = 0.0,
) -> None:
    """Write a minimal eval.csv with the columns the figure reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "env_name": [env_name] * n_rows,
        "cell": [cell] * n_rows,
        "algorithm": [algorithm] * n_rows,
        "behaviour_policy": [behaviour_policy] * n_rows,
        "seed": [seed] * n_rows,
        "alpha_conf": [alpha_conf] * n_rows,
        "step": list(range(1, n_rows + 1)),
        "eval_return_mean": [base_eval + 0.01 * i for i in range(n_rows)],
        "eval_oracle_return_mean": [base_oracle] * n_rows,
    })
    df.to_csv(path, index=False)


def _build_synthetic_results(
    root: Path,
    *,
    cells: list[int],
    algorithms: list[str],
    seeds: list[int],
    n_rows: int = 5,
) -> Path:
    results = root / "results"
    for cell in cells:
        for algo in algorithms:
            for seed in seeds:
                run_dir = results / f"cell{cell}_{algo}_seed{seed}"
                _write_eval_csv(
                    run_dir / "eval.csv",
                    env_name="bandit-tabular-sepsis-v0",
                    cell=cell,
                    algorithm=algo,
                    seed=seed,
                    n_rows=n_rows,
                )
    return results


def test_bandit_regret_renders_with_synthetic_runs(tmp_path: Path) -> None:
    """The figure produces a non-empty PDF on a 2 cells × 2 algos × 3 seeds
    fixture."""
    results = _build_synthetic_results(
        tmp_path,
        cells=[1, 3],
        algorithms=["ucb", "rct"],
        seeds=[0, 1, 2],
    )
    out = tmp_path / "out"
    make_bandit_regret(results, out)
    pdf = out / "bandit_regret.pdf"
    assert pdf.exists(), "bandit_regret.pdf was not written"
    assert pdf.stat().st_size > 1024, (
        f"bandit_regret.pdf is suspiciously small ({pdf.stat().st_size} bytes)"
    )


def test_bandit_regret_within_run_cumsum_is_monotone(tmp_path: Path) -> None:
    """v20 contract: cumulative regret is monotonic within each run.

    Pre-v20 the cumsum ran across the concatenated dataframe so values
    could decrease at run boundaries (whenever the next run's first
    eval row had a smaller regret_increment than the previous run's
    last cum_regret).  Post-v20 the cumsum is grouped by run identity
    so within-run monotonicity is guaranteed when increments are ≥ 0.
    """
    results = _build_synthetic_results(
        tmp_path,
        cells=[1, 3],
        algorithms=["ucb", "rct"],
        seeds=[0, 1],
        n_rows=6,
    )
    rows: list[pd.DataFrame] = []
    for run_dir in sorted(results.iterdir()):
        rows.append(pd.read_csv(run_dir / "eval.csv"))
    data = pd.concat(rows, ignore_index=True)
    out = _within_run_cum_regret(data)

    # Each (env_name, cell, algorithm, behaviour, seed, alpha_conf)
    # group should have a non-decreasing cum_regret along step.
    for _, g in out.groupby(_RUN_GROUPING):
        g_sorted = g.sort_values("step")
        cum = g_sorted["cum_regret"].to_numpy()
        diffs = np.diff(cum)
        assert (diffs >= -1e-9).all(), (
            f"cum_regret not monotone within run: increments {diffs.tolist()} "
            f"for run {g_sorted.iloc[0][_RUN_GROUPING].to_dict()}"
        )


def test_bandit_regret_skips_when_no_bandit_runs(tmp_path: Path) -> None:
    """If the results tree contains only non-bandit runs, the figure
    function should skip cleanly without raising and without writing
    a PDF."""
    results = tmp_path / "results"
    _write_eval_csv(
        results / "non_bandit_run" / "eval.csv",
        env_name="tabular-sepsis-v0",
        cell=1,
        algorithm="dqn",
        seed=0,
    )
    out = tmp_path / "out"
    make_bandit_regret(results, out)
    assert not (out / "bandit_regret.pdf").exists(), (
        "bandit_regret.pdf should not be written when no bandit runs are present"
    )
