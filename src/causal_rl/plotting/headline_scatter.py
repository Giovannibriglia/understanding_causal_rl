from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style, figure


def make_headline_scatter(
    results_dir: Path, output_dir: Path, env_prefix: str | None = None
) -> None:
    """Plot final gap vs oracle gap as mean points per (algorithm, behaviour).

    For each (algorithm, behaviour) compute the mean across runs/seeds of
    x = delta_tv and y = eval_oracle_return_mean - eval_return_mean.

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
            if env_prefix is not None and not str(last_row.get("env_name", "")).startswith(
                env_prefix
            ):
                continue
            rows.append(last_row)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    from causal_rl.plotting.colors import get_color, get_marker

    # Group by (algorithm, behaviour)
    groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in rows:
        try:
            algo = str(r.get("algorithm", "")).strip()
            beh = str(r.get("behaviour_policy", "")).strip()
            x = float(r["delta_tv"])
            y = float(r["eval_oracle_return_mean"]) - float(r["eval_return_mean"])
        except (ValueError, KeyError):
            continue
        groups.setdefault((algo, beh), []).append((x, y))

    fig, ax = figure("double", height_ratio=0.7)

    if not groups:
        ax.set_xlabel(r"Final $\widehat{\Delta}_{\mathrm{TV}}$")
        ax.set_ylabel("Return gap to oracle")
        pdf = output_dir / "headline_gap_vs_oracle.pdf"
        png = output_dir / "headline_gap_vs_oracle.png"
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
    for (algo, beh), _pts in groups.items():
        if beh not in beh_marker_map:
            beh_marker_map[beh] = get_marker(algo, beh) or "o"

    # Plot each group's seed points, ellipse and mean
    for (algo, beh), pts in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        pts_arr = np.array(pts, dtype=np.float64)
        mean_xy = pts_arr.mean(axis=0)
        std_xy = pts_arr.std(axis=0)
        # ellipse diameters = std_x, std_y (so radii = std/2)
        width = float(std_xy[0])
        height = float(std_xy[1])

        color = get_color(algo, beh)
        marker = get_marker(algo, beh) or "o"

        # faint individual seed points (behind ellipses)
        ax.scatter(pts_arr[:, 0], pts_arr[:, 1], color=color, alpha=0.25, s=24, zorder=0)
        all_seed_x.extend(pts_arr[:, 0].tolist())
        all_seed_y.extend(pts_arr[:, 1].tolist())

        # draw anisotropic ellipse for dispersion
        if width > 0 or height > 0:
            ell = Ellipse(
                (mean_xy[0], mean_xy[1]),
                width=width,
                height=height,
                facecolor=color,
                edgecolor=None,
                alpha=0.3,
                linewidth=0,
                zorder=1,
            )
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
    for algo, beh in sorted(groups.keys(), key=lambda x: (x[0] or "", x[1] or "")):
        color = get_color(algo, beh)
        marker = get_marker(algo, beh) or "o"
        label = f"{algo}" + (f" ({beh})" if beh else "")
        ln = Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="None",
            markersize=6,
            markeredgecolor="k",
        )
        combo_handles.append(ln)
        combo_labels.append(label)
    if combo_handles:
        ax.legend(
            combo_handles,
            combo_labels,
            title="Algorithm (behaviour)",
            loc="best",
            frameon=False,
            fontsize=7,
        )

    # Adjust axis limits to include seed points and ellipses
    if all_seed_x and all_seed_y and means:
        means_x = [m[0] for m in means]
        means_y = [m[1] for m in means]
        ellipse_lo_x = [mx - w / 2 for mx, w in zip(means_x, widths, strict=False)]
        ellipse_hi_x = [mx + w / 2 for mx, w in zip(means_x, widths, strict=False)]
        ellipse_lo_y = [my - h / 2 for my, h in zip(means_y, heights, strict=False)]
        ellipse_hi_y = [my + h / 2 for my, h in zip(means_y, heights, strict=False)]
        min_x = min(all_seed_x + ellipse_lo_x)
        max_x = max(all_seed_x + ellipse_hi_x)
        min_y = min(all_seed_y + ellipse_lo_y)
        max_y = max(all_seed_y + ellipse_hi_y)
        pad_x = (max_x - min_x) * 0.08 if max_x > min_x else 0.1
        pad_y = (max_y - min_y) * 0.08 if max_y > min_y else 0.1
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.set_xlabel(r"Final $\widehat{\Delta}_{\mathrm{TV}}$")
    ax.set_ylabel("Return gap to oracle")

    # Annotate the two characteristic clusters: low-Δ_TV / low-gap (id cells)
    # and high-Δ_TV / high-gap (partial_id cells).  Anchor positions inside
    # the current axis range so the labels remain visible across runs.
    if means:
        all_x = [m[0] for m in means]
        all_y = [m[1] for m in means]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        if max_x - min_x > 1e-6 and max_y - min_y > 1e-6:
            ax.text(
                min_x + 0.05 * (max_x - min_x),
                min_y + 0.10 * (max_y - min_y),
                "id cells\n(low Δ_TV, low gap)",
                fontsize=8,
                color="#2E7D32",
                ha="left",
                va="bottom",
            )
            ax.text(
                min_x + 0.65 * (max_x - min_x),
                min_y + 0.85 * (max_y - min_y),
                "partial_id cells\n(high Δ_TV, high gap)",
                fontsize=8,
                color="#E65100",
                ha="left",
                va="top",
            )

    pdf = output_dir / "headline_gap_vs_oracle.pdf"
    png = output_dir / "headline_gap_vs_oracle.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
