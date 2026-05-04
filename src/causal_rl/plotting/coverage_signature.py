"""Coverage-signature figure (v11).

Two panels of (ESS, min_propensity), log-log:

* Panel A — coloured by ``id_status``.  Healthy runs cluster top-right
  (ESS ≈ 1, min_propensity ≈ 0.1).  Coverage-broken runs from the
  ``coverage_stress`` sweep — when present in the manifest — cluster
  bottom-left (ESS < 0.5, min_propensity < 0.01).
* Panel B — same axes, coloured by ``gap_to_oracle``.  Reveals how
  coverage and gap correlate within the dataset.

Without ``coverage_stress`` data the two panels show essentially one
cluster (ESS ≈ 0.97); the figure becomes interesting only after the
v11 sweep is collected.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

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


def make_coverage_signature(
    results_dir: Path,
    output_dir: Path,
    env_prefix: str | None = None,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    points: list[dict[str, Any]] = []
    for path in results_dir.rglob("eval.csv"):
        last = _read_last_row(path)
        if last is None:
            continue
        if env_prefix is not None and not str(last.get("env_name", "")).startswith(
            env_prefix
        ):
            continue
        ess = _safe_float(last.get("ess_ratio"))
        min_prop = _safe_float(last.get("min_propensity"))
        eval_ret = _safe_float(last.get("eval_return_mean"))
        oracle_ret = _safe_float(last.get("eval_oracle_return_mean"))
        if not (np.isfinite(ess) and np.isfinite(min_prop)):
            continue
        gap = (
            (oracle_ret - eval_ret)
            if (np.isfinite(eval_ret) and np.isfinite(oracle_ret))
            else float("nan")
        )
        points.append(
            {
                "ess": max(1e-6, ess),
                "min_prop": max(1e-6, min_prop),
                "gap": gap,
                "id_status": str(last.get("id_status", "non_id")),
            }
        )
    if not points:
        print("[coverage_signature] no eval.csv rows found; skipping.")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Panel A: id_status colouring.
    for status, color in _ID_STATUS_COLORS.items():
        s_pts = [p for p in points if p["id_status"] == status]
        if not s_pts:
            continue
        ax_a.scatter(
            [p["ess"] for p in s_pts],
            [p["min_prop"] for p in s_pts],
            color=color,
            alpha=0.6,
            s=24,
            edgecolor="black",
            linewidth=0.2,
            label=status,
        )
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("ESS / N")
    ax_a.set_ylabel("min propensity")
    ax_a.set_title("Coverage signature, by id_status")
    ax_a.axvline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax_a.axhline(0.01, color="#888888", linestyle=":", linewidth=0.8)
    ax_a.text(
        0.5,
        ax_a.get_ylim()[1],
        "ESS=0.5",
        fontsize=7,
        color="#888888",
        ha="left",
        va="top",
    )
    ax_a.legend(title="id_status", loc="lower right", fontsize=7)

    # Panel B: gap colouring.  NaN gaps shown as grey.
    finite = [p for p in points if np.isfinite(p["gap"])]
    if finite:
        gaps = np.array([p["gap"] for p in finite], dtype=np.float64)
        sc = ax_b.scatter(
            [p["ess"] for p in finite],
            [p["min_prop"] for p in finite],
            c=gaps,
            cmap="viridis_r",
            alpha=0.7,
            s=24,
            edgecolor="black",
            linewidth=0.2,
        )
        cbar = fig.colorbar(sc, ax=ax_b)
        cbar.set_label("gap_to_oracle")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("ESS / N")
    ax_b.set_ylabel("min propensity")
    ax_b.set_title("Coverage signature, by gap_to_oracle")
    ax_b.axvline(0.5, color="#888888", linestyle=":", linewidth=0.8)
    ax_b.axhline(0.01, color="#888888", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    pdf = output_dir / "coverage_signature.pdf"
    png = output_dir / "coverage_signature.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
