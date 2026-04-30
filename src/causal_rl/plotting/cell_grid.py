from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from causal_rl.plotting.style import apply_style


def make_cell_grid(results_dir: Path, output_dir: Path, env_prefix: str | None = None) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for path in results_dir.rglob("eval.csv"):
        with path.open("r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    if env_prefix is not None:
        rows = [r for r in rows if str(r.get("env_name", "")).startswith(env_prefix)]
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 4.5), sharex=True, sharey=True)
    from causal_rl.plotting.colors import get_color, get_linestyle, get_marker

    combos_all = []
    seen = set()
    # collect all (algo,beh) combos present in the dataset for a complete legend
    for r in rows:
        algo_r = str(r.get("algorithm", "")).strip()
        beh_r = str(r.get("behaviour_policy", "")).strip()
        if (algo_r, beh_r) not in seen:
            seen.add((algo_r, beh_r))
            combos_all.append((algo_r, beh_r))

    # plot representative point per (algo,beh) for each cell to ensure visible color variety
    max_per_cell = 12
    for cell in range(1, 9):
        ax = axes[(cell - 1) // 4, (cell - 1) % 4]
        cell_rows = [r for r in rows if r.get("cell") == str(cell)]
        # group rows by (algo,beh); pick the row with the largest step as representative
        combo_map: dict[tuple[str, str], dict[str, str]] = {}
        for r in cell_rows:
            try:
                algo = str(r.get("algorithm", "")).strip()
                beh = str(r.get("behaviour_policy", "")).strip()
                step = int(r.get("step", 0))
            except (ValueError, KeyError):
                continue
            key = (algo, beh)
            prev = combo_map.get(key)
            if prev is None or step > int(prev.get("step", 0)):
                combo_map[key] = r
        # sort combos for stable plotting
        combo_items = sorted(combo_map.items(), key=lambda x: (x[0][0], x[0][1]))
        for idx, ((algo, beh), r) in enumerate(combo_items[:max_per_cell]):
            try:
                x = float(r["delta_tv"])
                y = float(r["eval_return_mean"])
            except (ValueError, KeyError):
                continue
            color = get_color(algo, beh)
            marker = get_marker(algo, beh) or "o"
            ax.scatter(x, y, s=40, color=color, marker=marker, edgecolor="k", linewidth=0.2)
            if idx % 4 == 0:
                ax.plot([x, x], [y - 0.02, y + 0.02], linewidth=0.6, color=color)
        ax.set_title(f"Cell {cell}")

    # create a legend for algorithm+behaviour mapping using all observed combos
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for algo, beh in combos_all:
        color = get_color(algo, beh)
        marker = get_marker(algo, beh) or "o"
        ls = get_linestyle(algo, beh)
        ln = Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=ls,
            markersize=6,
        )
        label = f"{algo}" + (f" ({beh})" if beh else "")
        handles.append(ln)
        labels.append(label)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(6, len(handles)),
            frameon=False,
            fontsize=8,
        )

    axes[1, 0].set_xlabel(r"$\widehat{\Delta}_{\mathrm{TV}}$")
    axes[1, 1].set_xlabel(r"$\widehat{\Delta}_{\mathrm{TV}}$")
    axes[0, 0].set_ylabel("Eval Return")
    axes[1, 0].set_ylabel("Eval Return")
    pdf = output_dir / "cell_grid_8x4.pdf"
    png = output_dir / "cell_grid_8x4.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
