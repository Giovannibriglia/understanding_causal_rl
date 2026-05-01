"""Bias/coverage sweep figure (Phase D.1).

Three subplots sharing the x-axis (bias_strength):
  Left:   delta_tv with shaded bootstrap CI.
  Middle: min_propensity and ess_ratio on twin axes.
  Right:  eval_oracle_return_mean - eval_return_mean, one line per cell.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

_CELL_COLORS = {1: "#4CAF50", 2: "#2196F3", 3: "#FF9800", 4: "#F44336"}


def make_bias_sweep(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Generate bias_sweep.pdf from run results."""
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data: {cell: {bias_strength: {metric: [values]}}}
    data: dict[int, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for eval_path in results_dir.rglob("eval.csv"):
        meta_path = eval_path.parent / "meta.json"
        if not meta_path.exists():
            continue
        with eval_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = rows[-1]
        try:
            cell = int(last.get("cell", 0))
            bias_strength = float(last.get("bias_strength", 1.0))
            delta_tv = float(last.get("delta_tv", 0.0))
            ci_lo = float(last.get("delta_tv_ci_lo", delta_tv))
            ci_hi = float(last.get("delta_tv_ci_hi", delta_tv))
            gap = float(last.get("eval_oracle_return_mean", 0.0)) - float(
                last.get("eval_return_mean", 0.0)
            )
            min_prop = float(last.get("min_propensity", 0.0))
            ess = float(last.get("ess_ratio", 0.0))
        except (ValueError, KeyError):
            continue
        d = data[cell][bias_strength]
        d["delta_tv"].append(delta_tv)
        d["ci_lo"].append(ci_lo)
        d["ci_hi"].append(ci_hi)
        d["gap"].append(gap)
        d["min_propensity"].append(min_prop)
        d["ess_ratio"].append(ess)

    if not data:
        return

    cells = sorted(data.keys())
    fig, (ax_tv, ax_prop, ax_gap) = plt.subplots(1, 3, figsize=(12, 4), sharey=False)

    for cell in cells:
        color = _CELL_COLORS.get(cell, "gray")
        bs_vals = sorted(data[cell].keys())
        tv_means = [float(np.mean(data[cell][bs]["delta_tv"])) for bs in bs_vals]
        ci_lo_m = [float(np.mean(data[cell][bs]["ci_lo"])) for bs in bs_vals]
        ci_hi_m = [float(np.mean(data[cell][bs]["ci_hi"])) for bs in bs_vals]
        gap_means = [float(np.mean(data[cell][bs]["gap"])) for bs in bs_vals]
        prop_means = [float(np.mean(data[cell][bs]["min_propensity"])) for bs in bs_vals]
        ess_means = [float(np.mean(data[cell][bs]["ess_ratio"])) for bs in bs_vals]

        x = np.array(bs_vals, dtype=np.float64)
        ax_tv.plot(x, tv_means, color=color, label=f"cell {cell}")
        ax_tv.fill_between(x, ci_lo_m, ci_hi_m, color=color, alpha=0.2)
        ax_gap.plot(x, gap_means, color=color, label=f"cell {cell}")
        ax_prop.plot(x, prop_means, color=color, linestyle="-", label=f"min_prop c{cell}")
        ax_prop.plot(x, ess_means, color=color, linestyle="--", label=f"ess c{cell}")

    ax_tv.set_xlabel("bias_strength")
    ax_tv.set_ylabel("Δ_TV")
    ax_tv.set_title("Gap metric vs bias strength")
    ax_tv.legend(fontsize=7)

    ax_prop.set_xlabel("bias_strength")
    ax_prop.set_ylabel("propensity metric")
    ax_prop.set_title("Coverage vs bias strength")
    ax_prop.legend(fontsize=6)

    ax_gap.set_xlabel("bias_strength")
    ax_gap.set_ylabel("Return gap to oracle")
    ax_gap.set_title("Final return gap vs bias strength")
    ax_gap.legend(fontsize=7)

    fig.suptitle("Bias / coverage sweep", fontsize=11)
    fig.tight_layout()
    pdf = output_dir / "bias_sweep.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
