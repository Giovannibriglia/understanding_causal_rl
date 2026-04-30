"""Sample-size sweep figure (Phase D.2).

x-axis: offline_transitions (log scale).
y-axis: final return gap to oracle.
One line per cell coloured by id_status. For partial_id cells, overlay
a horizontal dashed line at bound_width_mean / 2.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

_ID_STATUS_COLORS = {"id": "#4CAF50", "partial_id": "#FF9800", "non_id": "#F44336"}


def make_sample_sweep(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Generate sample_sweep.pdf from run results."""
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect: {cell: {n_transitions: {gap: float, bound_width_mean: float}}}
    data: dict[int, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    id_status_map: dict[int, str] = {}

    for eval_path in results_dir.rglob("eval.csv"):
        meta_path = eval_path.parent / "meta.json"
        if not meta_path.exists():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            n_trans = int(meta.get("offline_transitions", 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue

        with eval_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = rows[-1]
        try:
            cell = int(last.get("cell", 0))
            id_status = str(last.get("id_status", "non_id"))
            gap = float(last.get("eval_oracle_return_mean", 0.0)) - float(
                last.get("eval_return_mean", 0.0)
            )
            bwm = float(last.get("bound_width_mean", 0.0))
        except (ValueError, KeyError):
            continue
        data[cell][n_trans]["gap"].append(gap)
        data[cell][n_trans]["bound_width_mean"].append(bwm)
        id_status_map[cell] = id_status

    if not data:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    cells = sorted(data.keys())
    for cell in cells:
        id_status = id_status_map.get(cell, "non_id")
        color = _ID_STATUS_COLORS.get(id_status, "gray")
        ns_vals = sorted(data[cell].keys())
        gap_means = [float(np.mean(data[cell][n]["gap"])) for n in ns_vals]
        ax.plot(
            ns_vals,
            gap_means,
            color=color,
            marker="o",
            label=f"cell {cell} ({id_status})",
        )
        if id_status == "partial_id":
            bwm_vals = [float(np.mean(data[cell][n]["bound_width_mean"])) / 2 for n in ns_vals]
            ax.plot(
                ns_vals,
                bwm_vals,
                color=color,
                linestyle="--",
                alpha=0.6,
                label=f"bound floor c{cell}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("offline_transitions")
    ax.set_ylabel("Return gap to oracle")
    ax.set_title("Sample-size sweep")
    ax.legend(fontsize=7)
    fig.tight_layout()
    pdf = output_dir / "sample_sweep.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
