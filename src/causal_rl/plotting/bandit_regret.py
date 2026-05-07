from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RUN_GROUPING = ["env_name", "cell", "algorithm", "behaviour_policy", "seed", "alpha_conf"]


def _within_run_cum_regret(data: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative regret per run.

    Pre-v20 the previous implementation called ``cumsum`` on a
    dataframe concatenated across runs, summing across run boundaries
    and producing meaningless numbers (the apparent "RCT outperforms
    UCB" finding was a consequence of this bug).  v20 groups by run
    identity first, sorts by ``step`` within each group, then cumsums
    the per-step ``oracle - eval`` gap within the group.
    """
    data = data.sort_values(_RUN_GROUPING + ["step"])
    data["regret_increment"] = (
        data["eval_oracle_return_mean"] - data["eval_return_mean"]
    )
    data["cum_regret"] = data.groupby(_RUN_GROUPING)["regret_increment"].cumsum()
    return data


def make_bandit_regret(results_dir: Path, output_dir: Path) -> None:
    """Per-cell cumulative regret of bandit algorithms.

    v20: replaces the previous implementation, which had four bugs:
    identical Δ panels (no filtering on ``delta``), wrong cumsum unit
    (across runs not within), broken regret comparisons (RCT
    "outperforming" UCB as a consequence of the cumsum bug), and a
    fictional Δ axis the env doesn't actually parameterise on.

    The rewrite plots cumulative expected regret per training step,
    faceted by cell, with mean ± standard error across seeds.  v19's
    cell collapse means C1 ≡ C2 and C3 ≡ C4 in expectation, so two
    facets {C1, C3} are sufficient — they capture the meaningful
    ``expose_z`` axis without redundancy.
    """
    rows = []
    for run_dir in results_dir.glob("*"):
        eval_path = run_dir / "eval.csv"
        if not eval_path.exists():
            continue
        try:
            df = pd.read_csv(eval_path)
        except Exception:
            continue
        if df.empty or "env_name" not in df.columns:
            continue
        if not str(df["env_name"].iloc[0]).startswith("bandit-tabular-sepsis"):
            continue
        rows.append(df)
    if not rows:
        print("[bandit_regret] no bandit runs found; skipping.")
        return
    data = pd.concat(rows, ignore_index=True)
    if "eval_oracle_return_mean" not in data.columns or "eval_return_mean" not in data.columns:
        print("[bandit_regret] missing oracle/eval return columns; skipping.")
        return

    cells_to_plot = [c for c in [1, 3] if (data["cell"] == c).any()]
    if not cells_to_plot:
        print("[bandit_regret] no cells in {1, 3} present; skipping.")
        return

    data = _within_run_cum_regret(data)

    fig, axes = plt.subplots(
        1, len(cells_to_plot), figsize=(4.5 * len(cells_to_plot), 3.5), sharey=True
    )
    if len(cells_to_plot) == 1:
        axes = [axes]

    for ax, cell in zip(axes, cells_to_plot):
        cell_data = data[data["cell"] == cell]
        for algo, g in cell_data.groupby("algorithm"):
            agg = (
                g.groupby("step")["cum_regret"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.plot(agg["step"], agg["mean"], label=algo, linewidth=1.5)
            ax.fill_between(
                agg["step"],
                agg["mean"] - agg["sem"],
                agg["mean"] + agg["sem"],
                alpha=0.2,
            )
        cell_label = "C1 (expose_Z)" if cell == 1 else "C3 (Z hidden)"
        ax.set_title(cell_label)
        ax.set_xlabel("training step")
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("cumulative regret (oracle − agent)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Bandit cumulative regret by cell", fontsize=11)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "bandit_regret.pdf")
    plt.close(fig)
