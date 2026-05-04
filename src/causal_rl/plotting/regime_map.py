"""Regime-map: the diagnostic-framing headline figure (v11).

Each run is one point in the (Δ_TV, gap_to_oracle) plane.  Quadrant
labels and one-sentence diagnoses are overlaid:

  - low Δ_TV, low gap   → "Identifiable"
  - low Δ_TV, high gap  → "Coverage-broken"     (NEW with v11)
  - high Δ_TV, low gap  → "Confounding visible, recoverable"
  - high Δ_TV, high gap → "Partial observability"

Quadrant boundaries are computed from the data: ``x_thresh`` is the
median Δ_TV across id-cell runs (a healthy floor), ``y_thresh`` is the
median gap across runs whose ESS exceeds 0.9 (a coverage-healthy
baseline).  When the manifest contains ``coverage_stress`` runs the
coverage-broken quadrant gets populated; otherwise it stays empty.

This figure replaces ``headline_gap_vs_oracle.png`` as the headline.
The reader's task is no longer "look how well the regression fits" but
"find your regime and read the diagnosis".
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

_ID_STATUS_COLORS = {"id": "#4CAF50", "partial_id": "#FF9800", "non_id": "#F44336"}
_BEH_MARKERS = {
    "uniform": "o",
    "reward_aligned": "^",
    "reward_misaligned": "v",
    "information": "s",
    "certainty_seeking": "D",
    "curiosity": "X",
    "novelty_trap": "P",
}


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


def _quadrant_label(delta_tv: float, gap: float, x_thresh: float, y_thresh: float) -> str:
    if delta_tv <= x_thresh and gap <= y_thresh:
        return "Identifiable"
    if delta_tv <= x_thresh and gap > y_thresh:
        return "Coverage-broken"
    if delta_tv > x_thresh and gap <= y_thresh:
        return "Confounding visible, recoverable"
    return "Partial observability"


def make_regime_map(
    results_dir: Path,
    output_dir: Path,
    env_prefix: str | None = None,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, object]] = []
    for path in results_dir.rglob("eval.csv"):
        last = _read_last_row(path)
        if last is None:
            continue
        if env_prefix is not None and not str(last.get("env_name", "")).startswith(
            env_prefix
        ):
            continue
        delta_tv = _safe_float(last.get("delta_tv"))
        eval_ret = _safe_float(last.get("eval_return_mean"))
        oracle_ret = _safe_float(last.get("eval_oracle_return_mean"))
        ess = _safe_float(last.get("ess_ratio"))
        if not (np.isfinite(delta_tv) and np.isfinite(eval_ret) and np.isfinite(oracle_ret)):
            continue
        gap = oracle_ret - eval_ret
        points.append(
            {
                "delta_tv": delta_tv,
                "gap": gap,
                "ess": ess,
                "id_status": str(last.get("id_status", "non_id")),
                "behaviour": str(last.get("behaviour_policy", "uniform")),
            }
        )
    if not points:
        print("[regime_map] no eval.csv rows found; skipping.")
        return

    # Quadrant thresholds.  x: median Δ_TV across id-cell runs.  y: median
    # gap across coverage-healthy (ESS > 0.9) runs.
    id_dtv = [p["delta_tv"] for p in points if p["id_status"] == "id"]
    healthy_gaps = [p["gap"] for p in points if isinstance(p["ess"], float) and p["ess"] > 0.9]
    x_thresh = float(np.median(id_dtv)) if id_dtv else 0.04
    y_thresh = float(np.median(healthy_gaps)) if healthy_gaps else 5.0

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    # Quadrant background bands.
    xlim_lo = float(min(p["delta_tv"] for p in points))
    xlim_hi = float(max(p["delta_tv"] for p in points))
    ylim_lo = float(min(p["gap"] for p in points))
    ylim_hi = float(max(p["gap"] for p in points))
    pad_x = 0.08 * max(1e-6, xlim_hi - xlim_lo)
    pad_y = 0.08 * max(1e-6, ylim_hi - ylim_lo)
    ax.set_xlim(xlim_lo - pad_x, xlim_hi + pad_x)
    ax.set_ylim(ylim_lo - pad_y, ylim_hi + pad_y)

    quadrant_colors = {
        "Identifiable": "#E8F5E9",
        "Coverage-broken": "#FFEBEE",
        "Confounding visible, recoverable": "#FFF8E1",
        "Partial observability": "#FCE4EC",
    }
    # Draw the four quadrant tints.
    ax.axvspan(xlim_lo - pad_x, x_thresh, ymin=0, ymax=1, alpha=0)  # placeholder
    ax.fill_between(
        [xlim_lo - pad_x, x_thresh],
        ylim_lo - pad_y,
        y_thresh,
        color=quadrant_colors["Identifiable"],
        alpha=1,
        zorder=0,
    )
    ax.fill_between(
        [xlim_lo - pad_x, x_thresh],
        y_thresh,
        ylim_hi + pad_y,
        color=quadrant_colors["Coverage-broken"],
        alpha=1,
        zorder=0,
    )
    ax.fill_between(
        [x_thresh, xlim_hi + pad_x],
        ylim_lo - pad_y,
        y_thresh,
        color=quadrant_colors["Confounding visible, recoverable"],
        alpha=1,
        zorder=0,
    )
    ax.fill_between(
        [x_thresh, xlim_hi + pad_x],
        y_thresh,
        ylim_hi + pad_y,
        color=quadrant_colors["Partial observability"],
        alpha=1,
        zorder=0,
    )
    ax.axvline(x_thresh, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(y_thresh, color="#888888", linestyle="--", linewidth=0.8, zorder=1)

    # Scatter points coloured by id_status, shaped by behaviour.
    for p in points:
        color = _ID_STATUS_COLORS.get(p["id_status"], "gray")
        marker = _BEH_MARKERS.get(p["behaviour"], "o")
        ax.scatter(
            p["delta_tv"],
            p["gap"],
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.3,
            s=36,
            alpha=0.7,
            zorder=2,
        )

    # Quadrant labels in the corners.
    text_kwargs = {
        "fontsize": 8,
        "fontweight": "bold",
        "transform": ax.transAxes,
        "bbox": {
            "boxstyle": "round,pad=0.25",
            "fc": "white",
            "ec": "#444444",
            "alpha": 0.85,
        },
    }
    ax.text(0.02, 0.02, "Identifiable\n(low Δ_TV, low gap)", ha="left", va="bottom", color="#1B5E20", **text_kwargs)
    ax.text(0.02, 0.98, "Coverage-broken\n(low Δ_TV, high gap)", ha="left", va="top", color="#B71C1C", **text_kwargs)
    ax.text(0.98, 0.02, "Confounding visible,\nrecoverable", ha="right", va="bottom", color="#E65100", **text_kwargs)
    ax.text(0.98, 0.98, "Partial observability\n(high Δ_TV, high gap)", ha="right", va="top", color="#880E4F", **text_kwargs)

    # id_status legend.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=col,
            marker="o",
            linestyle="None",
            markersize=7,
            markeredgecolor="black",
            label=status,
        )
        for status, col in _ID_STATUS_COLORS.items()
    ]
    ax.legend(
        handles=legend_handles,
        title="id_status",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
    )
    ax.set_xlabel(r"Final $\widehat{\Delta}_{\mathrm{TV}}$")
    ax.set_ylabel("Return gap to oracle")
    ax.set_title(
        f"Regime map  ·  thresholds: Δ_TV={x_thresh:.3f}, gap={y_thresh:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    pdf = output_dir / "regime_map.pdf"
    png = output_dir / "regime_map.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def regime_quadrant_counts(
    results_dir: Path, env_prefix: str | None = None
) -> dict[str, int]:
    """Convenience: count how many runs fall in each quadrant.

    Used by the v11 acceptance check ("≥ 3 of 4 quadrants populated") and
    by ``summary.json``'s ``claim_6``.
    """
    counts: dict[str, int] = defaultdict(int)
    points: list[tuple[float, float]] = []
    id_dtv: list[float] = []
    healthy_gaps: list[float] = []
    for path in results_dir.rglob("eval.csv"):
        last = _read_last_row(path)
        if last is None:
            continue
        if env_prefix is not None and not str(last.get("env_name", "")).startswith(
            env_prefix
        ):
            continue
        delta_tv = _safe_float(last.get("delta_tv"))
        eval_ret = _safe_float(last.get("eval_return_mean"))
        oracle_ret = _safe_float(last.get("eval_oracle_return_mean"))
        ess = _safe_float(last.get("ess_ratio"))
        if not (np.isfinite(delta_tv) and np.isfinite(eval_ret) and np.isfinite(oracle_ret)):
            continue
        gap = oracle_ret - eval_ret
        points.append((delta_tv, gap))
        if str(last.get("id_status", "")) == "id":
            id_dtv.append(delta_tv)
        if np.isfinite(ess) and ess > 0.9:
            healthy_gaps.append(gap)
    if not points:
        return dict(counts)
    x_thresh = float(np.median(id_dtv)) if id_dtv else 0.04
    y_thresh = float(np.median(healthy_gaps)) if healthy_gaps else 5.0
    for delta_tv, gap in points:
        counts[_quadrant_label(delta_tv, gap, x_thresh, y_thresh)] += 1
    return dict(counts)
