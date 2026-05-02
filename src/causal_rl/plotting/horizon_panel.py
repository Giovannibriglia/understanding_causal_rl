"""Two-subplot horizon-impact panel.

Left: normalised eval return (eval / oracle) vs horizon, one line per cell.
Right: delta_tv vs horizon, one line per cell.

Reads ``summary.json``'s ``per_cell_per_horizon`` block produced by the matrix
runner when ``horizon_sweep`` is active.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from causal_rl.plotting.style import apply_style

_CELL_LABELS = {
    1: "C1 (id, expose_z, π_b known)",
    2: "C2 (expose_z, π_b unknown)",
    3: "C3 (no z, π_b known)",
    4: "C4 (no z, π_b unknown)",
}


def render_horizon_panel(
    summary: dict[str, Any],
    output_path: Path,
) -> Figure:
    """Render the horizon-impact panel.

    Args:
        summary: Parsed ``summary.json``.  Must contain ``per_cell_per_horizon``.
        output_path: Where to write the PDF.

    Returns:
        The matplotlib :class:`Figure` (after writing it to disk).
    """
    apply_style()

    per_cell_per_horizon = summary.get("per_cell_per_horizon") or {}
    if not per_cell_per_horizon:
        raise ValueError("summary.json missing per_cell_per_horizon block")

    horizons: set[int] = set()
    cells: set[int] = set()
    for key in per_cell_per_horizon:
        cell_str, h_str = key.split("_h", 1)
        cells.add(int(cell_str))
        horizons.add(int(h_str))
    horizons_sorted = sorted(horizons)
    cells_sorted = sorted(cells)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for cell in cells_sorted:
        norm_returns: list[float] = []
        delta_tvs: list[float] = []
        xs: list[int] = []
        for h in horizons_sorted:
            entry = per_cell_per_horizon.get(f"{cell}_h{h}")
            if not entry:
                continue
            eval_ret = entry.get("eval_return_mean")
            oracle_ret = entry.get("eval_oracle_return_mean")
            d_tv = entry.get("delta_tv_mean")
            if (
                eval_ret is None
                or oracle_ret is None
                or oracle_ret in (0.0, 0)
            ):
                continue
            norm_returns.append(float(eval_ret) / float(oracle_ret))
            delta_tvs.append(float(d_tv) if d_tv is not None else float("nan"))
            xs.append(h)
        if not xs:
            continue
        label = _CELL_LABELS.get(cell, f"cell {cell}")
        ax_left.plot(xs, norm_returns, marker="o", label=label)
        ax_right.plot(xs, delta_tvs, marker="o", label=label)

    ax_left.set_xlabel("horizon (episode length)")
    ax_left.set_ylabel("normalised return (eval / oracle)")
    ax_left.set_title("Return quality vs horizon")
    ax_left.set_xscale("log")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc="best", fontsize=8)

    ax_right.set_xlabel("horizon (episode length)")
    ax_right.set_ylabel(r"$\Delta_{\mathrm{TV}}$")
    ax_right.set_title(r"$\Delta_{\mathrm{TV}}$ vs horizon")
    ax_right.set_xscale("log")
    ax_right.grid(True, alpha=0.3)

    fig.suptitle("Horizon impact: id strata stay flat; partial_id strata diverge")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def render_horizon_panel_from_summary_path(
    summary_path: Path,
    output_path: Path,
) -> Figure:
    """Convenience: load ``summary.json`` from disk and render."""
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return render_horizon_panel(summary, output_path)


__all__ = [
    "render_horizon_panel",
    "render_horizon_panel_from_summary_path",
]
