from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style, figure


def make_headline_scatter(results_dir: Path, output_dir: Path, env_prefix: str | None = None) -> None:
    """Plot final gap vs generalisation as mean points per (algorithm, behaviour).

    For each (algorithm, behaviour) compute the mean across runs/seeds of
    x = delta_tv and y = eval_return_mean - eval_holdout_return_mean.

    Plot faint individual seed points, then an anisotropic ellipse (width=std_x,
    height=std_y) with alpha=0.3 to visualise dispersion, and the mean marker on
    top. Two legends are shown: algorithm (colour) and behaviour (marker).
    """
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    # Use only the final evaluation row from each eval.csv to avoid plotting
    # every checkpoint (reduces clutter: one point per run/file instead of many).
    for path in results_dir.rglob("eval.csv"):
        with path.open("r", encoding="utf-8") as f:
            rdr = list(csv.DictReader(f))
            if not rdr:
                continue
            last_row = rdr[-1]
            if env_prefix is not None and not str(last_row.get("env_name", "")).startswith(env_prefix):
                continue
            rows.append(last_row)

    from causal_rl.plotting.colors import get_style
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    # Group by (algorithm, behaviour)
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        try:
            algo = str(r.get("algorithm", "")).strip()
            beh = str(r.get("behaviour_policy", "")).strip()
            x = float(r["delta_tv"])
            y = float(r["eval_return_mean"]) - float(r["eval_holdout_return_mean"])
        except (ValueError, KeyError):
            continue
        groups.setdefault((algo, beh), []).append((x, y))

    fig, ax = figure("double", height_ratio=0.7)

    if not groups:
        ax.set_xlabel(r"Final $\widehat{\Delta}_{\mathrm{TV}}$")
        ax.set_ylabel("Train - Holdout Return")
        pdf = output_dir / "headline_gap_vs_generalisation.pdf"
        png = output_dir / "headline_gap_vs_generalisation.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    means: list[tuple[float, float]] = []
    widths: list[float] = []
    heights: list[float] = []

    # Collect for axis autoscaling including seed points
    all_seed_x: list[float] = []
    all_seed_y: list[float] = []

    algo_set: set[str] = set()
    beh_set: set[str] = set()

    # Precompute behaviour -> marker mapping (marker depends only on behaviour)
    beh_marker_map: dict[str, str] = {}
    for (algo, beh), pts in groups.items():
        if beh not in beh_marker_map:
            style = get_style(algo, beh)
            beh_marker_map[beh] = style.get("marker") or "o"

    # Plot each group's seed points, ellipse and mean
    for (algo, beh), pts in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        pts_arr = np.array(pts, dtype=np.float64)
        mean_xy = pts_arr.mean(axis=0)
        std_xy = pts_arr.std(axis=0)
        # ellipse diameters = std_x, std_y (so radii = std/2)
        width = float(std_xy[0])
        height = float(std_xy[1])

        style = get_style(algo, beh)
        color = style.get("color")
        marker = style.get("marker") if style.get("marker") is not None else "o"

        # faint individual seed points (behind ellipses)
        ax.scatter(pts_arr[:, 0], pts_arr[:, 1], color=color, alpha=0.25, s=24, zorder=0)
        all_seed_x.extend(pts_arr[:, 0].tolist())
        all_seed_y.extend(pts_arr[:, 1].tolist())

        # draw anisotropic ellipse for dispersion
        if width > 0 or height > 0:
            ell = Ellipse((mean_xy[0], mean_xy[1]), width=width, height=height,
                          facecolor=color, edgecolor=None, alpha=0.3, linewidth=0, zorder=1)
            ax.add_patch(ell)

        # plot mean point on top
        ax.scatter(mean_xy[0], mean_xy[1], color=color, marker=marker, edgecolor="k", zorder=3)

        means.append((mean_xy[0], mean_xy[1]))
        widths.append(width)
        heights.append(height)

        algo_set.add(algo)
        beh_set.add(beh)

    # Single legend: one entry per (algorithm, behaviour)
    combo_handles = []
    combo_labels = []
    for (algo, beh) in sorted(groups.keys(), key=lambda x: (x[0] or "", x[1] or "")):
        style = get_style(algo, beh)
        color = style.get("color")
        marker = style.get("marker") or "o"
        label = f"{algo}" + (f" ({beh})" if beh else "")
        ln = Line2D([0], [0], color=color, marker=marker, linestyle="None", markersize=6, markeredgecolor="k")
        combo_handles.append(ln)
        combo_labels.append(label)
    if combo_handles:
        ax.legend(combo_handles, combo_labels, title="Algorithm (behaviour)", loc="best", frameon=False, fontsize=7)

    # Adjust axis limits to include seed points and ellipses
    if all_seed_x and all_seed_y and means:
        min_x = min(all_seed_x + [mx - w / 2 for mx, w in zip([m[0] for m in means], widths)])
        max_x = max(all_seed_x + [mx + w / 2 for mx, w in zip([m[0] for m in means], widths)])
        min_y = min(all_seed_y + [my - h / 2 for my, h in zip([m[1] for m in means], heights)])
        max_y = max(all_seed_y + [my + h / 2 for my, h in zip([m[1] for m in means], heights)])
        pad_x = (max_x - min_x) * 0.08 if max_x > min_x else 0.1
        pad_y = (max_y - min_y) * 0.08 if max_y > min_y else 0.1
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.set_xlabel(r"Final $\widehat{\Delta}_{\mathrm{TV}}$")
    ax.set_ylabel("Train - Holdout Return")

    pdf = output_dir / "headline_gap_vs_generalisation.pdf"
    png = output_dir / "headline_gap_vs_generalisation.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
