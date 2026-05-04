"""Three-panel summary of the v8 causal-RL metrics, stratified by cell.

Each panel addresses a different failure mode:

* Panel 1 — ``backdoor_residual_mean``: only meaningful in cells with
  observed Z (1, 2).  Should converge to zero with sample size if the
  back-door criterion is satisfied; stays non-zero otherwise.
* Panel 2 — ``cond_mi_r_z_given_sa``: all four cells.  Higher in
  partial-observability cells (3, 4) where Z carries reward-relevant
  information beyond what S and A explain.
* Panel 3 — ``target_support_overlap``: all four cells.  Indicates
  whether the eval-distribution support is contained in π_b's support;
  values < 0.5 signal off-policy estimation is unreliable.

Each panel is a per-cell box plot.  Cells are ordered 1, 2, 3, 4 and
coloured by ``id_status``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

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


def make_causal_metrics_grid(
    results_dir: Path,
    output_dir: Path,
    env_prefix: str | None = None,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    # per_cell[cell][metric] = list[float] (NaNs filtered out)
    per_cell: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cell_id_status: dict[int, str] = {}
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
        cell_id_status.setdefault(cell, str(last.get("id_status", "non_id")))
        for col in (
            "backdoor_residual_mean",
            "cond_mi_r_z_given_sa",
            "target_support_overlap",
        ):
            v = _safe_float(last.get(col))
            if np.isfinite(v):
                per_cell[cell][col].append(v)
    if not per_cell:
        print("[causal_metrics_grid] no eval.csv rows found; skipping.")
        return

    cells = sorted(per_cell.keys())
    panels = (
        ("backdoor_residual_mean", "Backdoor residual", "id-cells should ↓0; partial_id stays >0"),
        ("cond_mi_r_z_given_sa", r"$I(R; Z \mid S, A)$", "Higher in partial-observability cells (3, 4)"),
        ("target_support_overlap", "Target support overlap", "Values < 0.5 signal unreliable off-policy estimation"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for ax, (col, title, subtitle) in zip(axes, panels, strict=True):
        data: list[list[float]] = []
        labels: list[str] = []
        colors: list[str] = []
        for cell in cells:
            vals = per_cell[cell].get(col, [])
            if vals:
                data.append(vals)
                status = cell_id_status.get(cell, "non_id")
                labels.append(f"C{cell}\n({status})")
                colors.append(_ID_STATUS_COLORS.get(status, "gray"))
        if not data:
            ax.text(
                0.5,
                0.5,
                "not collected",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#888888",
                style="italic",
            )
            ax.set_title(title, fontsize=9)
            continue
        # ``tick_labels`` is the post-3.9 spelling; fall back to ``labels``
        # for older matplotlib.
        try:
            bp = ax.boxplot(
                data,
                tick_labels=labels,
                patch_artist=True,
                medianprops={"color": "black"},
            )
        except TypeError:
            bp = ax.boxplot(
                data,
                labels=labels,
                patch_artist=True,
                medianprops={"color": "black"},
            )
        for patch, color in zip(bp["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(subtitle, fontsize=7, color="#555555", style="italic")
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle("Causal-RL diagnostic metrics by cell", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf = output_dir / "causal_metrics_grid.pdf"
    png = output_dir / "causal_metrics_grid.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
