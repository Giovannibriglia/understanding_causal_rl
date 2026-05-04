"""Five-divergence robustness panel.

The paper's primary metric is ``Δ_TV``.  Four others — KL, χ², sup, MMD²
and KS — measure the same gap ``‖P̂(R | a, s) − P̂(R | do(a), s)‖`` under
different choices of divergence.  This panel shows them side-by-side at
the final eval checkpoint, stratified by ``id_status``, so a reader can
verify the qualitative pattern is robust to the choice of divergence.

Layout: 4 panels (one per cell).  Each panel is a grouped bar chart with
divergences on the x-axis and id_status as the colour grouping.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

DIVERGENCES: tuple[tuple[str, str], ...] = (
    ("delta_tv", r"$\widehat{\Delta}_{\mathrm{TV}}$"),
    ("delta_kl", r"$\widehat{\Delta}_{\mathrm{KL}}$"),
    ("delta_chi2", r"$\widehat{\Delta}_{\chi^2}$"),
    ("delta_sup", r"$\widehat{\Delta}_{\sup}$"),
    ("delta_mmd2", r"$\widehat{\Delta}_{\mathrm{MMD}^2}$"),
    ("delta_ks", r"$\widehat{\Delta}_{\mathrm{KS}}$"),
)
_ID_STATUS_COLORS = {"id": "#4CAF50", "partial_id": "#FF9800", "non_id": "#F44336"}


def _safe_float(val: object) -> float:
    if val is None or val == "" or val == "nan":
        return float("nan")
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _read_last_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def make_divergence_panel(
    results_dir: Path,
    output_dir: Path,
    env_prefix: str | None = None,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    # values[cell][id_status][divergence] = list[float]
    values: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for path in results_dir.rglob("eval.csv"):
        last = _read_last_row(path)
        if last is None:
            continue
        if env_prefix is not None and not str(last.get("env_name", "")).startswith(
            env_prefix
        ):
            continue
        try:
            cell = int(last.get("cell", 0))
        except (TypeError, ValueError):
            continue
        id_status = str(last.get("id_status", "non_id"))
        for col, _ in DIVERGENCES:
            v = _safe_float(last.get(col))
            if np.isfinite(v):
                values[cell][id_status][col].append(v)
    if not values:
        print("[divergence_panel] no eval.csv rows found; skipping.")
        return

    cells_present = sorted(values.keys())
    fig, axes = plt.subplots(
        1, len(cells_present), figsize=(3 * max(1, len(cells_present)), 3.5), sharey=True
    )
    if len(cells_present) == 1:
        axes = [axes]
    n_div = len(DIVERGENCES)
    width = 0.25
    statuses_global = ("id", "partial_id", "non_id")
    x_positions = np.arange(n_div)
    for ax, cell in zip(axes, cells_present, strict=True):
        for offset_idx, status in enumerate(statuses_global):
            means = []
            for col, _ in DIVERGENCES:
                vals = values[cell].get(status, {}).get(col, [])
                means.append(float(np.mean(vals)) if vals else 0.0)
            offset = (offset_idx - 1) * width
            ax.bar(
                x_positions + offset,
                means,
                width=width,
                label=status,
                color=_ID_STATUS_COLORS.get(status, "gray"),
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [name for _, name in DIVERGENCES], rotation=30, ha="right", fontsize=7
        )
        ax.set_title(f"Cell {cell}", fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("divergence value (final eval)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="id_status",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(handles),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Five-divergence robustness check (final checkpoint)", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf = output_dir / "divergence_panel.pdf"
    png = output_dir / "divergence_panel.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
