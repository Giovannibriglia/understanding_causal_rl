from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_bound_width_panel(results_dir: Path, output_dir: Path) -> None:
    rows = []
    for run_dir in results_dir.glob("*"):
        eval_path = run_dir / "eval.csv"
        if not eval_path.exists():
            continue
        try:
            df = pd.read_csv(eval_path)
        except Exception:
            continue
        if df.empty:
            continue
        rows.append(df.sort_values("step").iloc[-1])
    if not rows:
        print("[bound_width_panel] no eval rows found; skipping.")
        return
    data = pd.DataFrame(rows)
    if "bound_width_mean" not in data.columns or "eval_oracle_return_mean" not in data.columns:
        print("[bound_width_panel] required columns missing; skipping.")
        return
    x = data["bound_width_mean"]
    y = data["eval_oracle_return_mean"] - data["eval_return_mean"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("bound_width_mean")
    ax.set_ylabel("final return gap")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "bound_width_panel.pdf")
    plt.close(fig)
