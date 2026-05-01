from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_bandit_regret(
    results_dir: Path,
    output_dir: Path,
    deltas: list[float] = [0.05, 0.1, 0.2, 0.3, 0.4],
) -> None:
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
    fig, axes = plt.subplots(1, len(deltas), figsize=(3 * len(deltas), 3), sharey=True)
    if len(deltas) == 1:
        axes = [axes]
    for ax, delta in zip(axes, deltas):
        for algo, g in data.groupby("algorithm"):
            gg = g.sort_values("step")
            if "eval_oracle_return_mean" not in gg.columns:
                continue
            regret = (gg["eval_oracle_return_mean"] - gg["eval_return_mean"]).cumsum()
            ax.plot(gg["step"], regret, label=algo)
        ax.set_title(f"Δ={delta}")
        ax.set_xlabel("episode")
    axes[0].set_ylabel("cumulative regret")
    axes[0].legend(fontsize=7)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "bandit_regret.pdf")
    plt.close(fig)
