"""Identifiability panel: final return gap grouped by id_status.

For each algorithm, plots the final return gap to the oracle (eval_oracle_return_mean
- eval_return_mean) with cells grouped on the x-axis by their id_status.

Expected qualitative pattern:
  - ``non_id`` cells: large gap, insensitive to sample size
  - ``partial_id`` cells: moderate gap, bounded by Bareinboim bounds
  - ``id`` cells: gap approaches zero with sufficient coverage
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.identification.id_oracle import ID_STATUS_ORDER
from causal_rl.plotting.style import apply_style

_ID_STATUS_LABELS = {
    "id": "Identifiable",
    "partial_id": "Partially\nidentifiable",
    "non_id": "Non-\nidentifiable",
}

_ID_STATUS_COLORS = {
    "id": "#4CAF50",  # green
    "partial_id": "#FF9800",  # orange
    "non_id": "#F44336",  # red
}


def make_identifiability_panel(
    results_dir: Path,
    output_dir: Path,
    env_prefix: str | None = None,
) -> None:
    """Plot return gap to oracle, grouped by id_status, one subplot per algorithm."""
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for path in results_dir.rglob("eval.csv"):
        with path.open("r", encoding="utf-8") as f:
            rdr = list(csv.DictReader(f))
            if not rdr:
                continue
            last = rdr[-1]
            if env_prefix is not None and not str(last.get("env_name", "")).startswith(env_prefix):
                continue
            rows.append(last)

    if not rows:
        return

    # Collect algorithms and build data structure:
    # {algo: {id_status: [gap_value, ...]}}
    data: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        try:
            algo = str(r.get("algorithm", "")).strip()
            id_status = str(r.get("id_status", "non_id")).strip()
            oracle = float(r["eval_oracle_return_mean"])
            ret = float(r["eval_return_mean"])
            gap = oracle - ret
        except (ValueError, KeyError):
            continue
        data.setdefault(algo, {}).setdefault(id_status, []).append(gap)

    if not data:
        return

    algos = sorted(data.keys())
    id_statuses = sorted(["id", "partial_id", "non_id"], key=lambda s: ID_STATUS_ORDER.get(s, 9))
    n_algos = len(algos)
    fig, axes = plt.subplots(1, n_algos, figsize=(4 * n_algos, 4), sharey=True)
    if n_algos == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos, strict=False):
        algo_data = data[algo]
        x_positions = []
        x_labels = []
        for i, status in enumerate(id_statuses):
            vals = algo_data.get(status, [])
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            color = _ID_STATUS_COLORS.get(status, "gray")
            ax.boxplot(
                arr,
                positions=[i],
                widths=0.6,
                patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.7},
                medianprops={"color": "black", "linewidth": 2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
            )
            ax.scatter(
                [i] * len(arr),
                arr,
                color=color,
                alpha=0.3,
                s=18,
                zorder=0,
            )
            x_positions.append(i)
            x_labels.append(_ID_STATUS_LABELS.get(status, status))

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(algo, fontsize=9)
        ax.set_xlabel("Identifiability status", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Return gap to oracle", fontsize=8)
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.suptitle("Return gap to oracle by identifiability status", fontsize=10)
    fig.tight_layout()

    pdf = output_dir / "identifiability_panel.pdf"
    png = output_dir / "identifiability_panel.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
