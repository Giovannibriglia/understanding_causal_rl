"""Per-arm bound-width vs per-arm gap-to-optimal scatter.

Each marker is one ``(run × arm)`` point.  Bound width on x; gap-to-optimal
on y.  The point of the figure is to show that informative bounds (left of
``1 - 1/n_actions``) correspond to over-played arms (small gap) and that
*uninformative* bounds (≈ ``1 - 1/n_actions``) tell us the offline data
didn't help — those are the arms whose true reward could be anywhere.

Coloured by ``id_status``, marker shape by behaviour family, with a
per-stratum regression line whose ``R²`` is shown in the legend.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

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
_N_ACTIONS = 8


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _read_last_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def make_bound_width_panel(results_dir: Path, output_dir: Path) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    points: list[dict[str, Any]] = []
    for eval_path in results_dir.rglob("eval.csv"):
        last = _read_last_row(eval_path)
        if last is None:
            continue
        id_status = str(last.get("id_status", "non_id"))
        behaviour = str(last.get("behaviour_policy", "uniform"))
        # Per-arm widths and per-arm μ̂ (= lower / max(p_a, 1e-6)).
        per_arm_lower = [_safe_float(last.get(f"bound_lower_a{a}")) for a in range(_N_ACTIONS)]
        per_arm_upper = [_safe_float(last.get(f"bound_upper_a{a}")) for a in range(_N_ACTIONS)]
        per_arm_count = [_safe_float(last.get(f"obs_arm_count_a{a}")) for a in range(_N_ACTIONS)]
        total = sum(c for c in per_arm_count if not np.isnan(c)) or 1.0
        # Estimated μ̂_a = E[R|A=a] in [0, 1] ≈ lower_a / p_a.
        mu_hats: list[float] = []
        for low, count in zip(per_arm_lower, per_arm_count, strict=True):
            p_a = count / total if total > 0 else 0.0
            mu_hats.append(low / p_a if p_a > 0.0 else float("nan"))
        clean_mu = [m for m in mu_hats if not np.isnan(m)]
        if not clean_mu:
            continue
        mu_star = max(clean_mu)
        for low, hi, m in zip(per_arm_lower, per_arm_upper, mu_hats, strict=True):
            if np.isnan(low) or np.isnan(hi) or np.isnan(m):
                continue
            width = hi - low
            gap = mu_star - m
            points.append(
                {
                    "width": width,
                    "gap": gap,
                    "id_status": id_status,
                    "behaviour": behaviour,
                }
            )
    if not points:
        print("[bound_width_panel] no per-arm bound rows found; skipping.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    # Draw guideline at the uniform-policy width.
    uniform_width = 1.0 - 1.0 / float(_N_ACTIONS)
    ax.axvline(uniform_width, linestyle="--", color="gray", alpha=0.6, linewidth=1)
    ax.text(
        uniform_width + 0.005,
        0.0,
        "uniform π_b\nfloor",
        fontsize=7,
        color="gray",
        ha="left",
        va="bottom",
    )

    # Plot per-stratum points and regression line.
    legend_entries: list[tuple[str, str]] = []
    for status, color in _ID_STATUS_COLORS.items():
        s_points = [p for p in points if p["id_status"] == status]
        if not s_points:
            continue
        for p in s_points:
            marker = _BEH_MARKERS.get(p["behaviour"], "o")
            ax.scatter(
                p["width"],
                p["gap"],
                color=color,
                marker=marker,
                alpha=0.5,
                s=24,
                edgecolor="k",
                linewidth=0.2,
            )
        xs = np.array([p["width"] for p in s_points], dtype=float)
        ys = np.array([p["gap"] for p in s_points], dtype=float)
        if len(xs) >= 3 and np.std(xs) > 1e-6:
            m, b = np.polyfit(xs, ys, 1)
            xs_line = np.linspace(xs.min(), xs.max(), 50)
            ax.plot(xs_line, m * xs_line + b, color=color, linewidth=1.5)
            ss_res = float(np.sum((ys - (m * xs + b)) ** 2))
            ss_tot = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
            legend_entries.append((status, f"{status} (n={len(xs)}, R²={r2:.2f})"))
        else:
            legend_entries.append((status, f"{status} (n={len(xs)})"))

    # Custom legend with id_status colours and behaviour markers.
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for status, label in legend_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                color=_ID_STATUS_COLORS[status],
                marker="o",
                linestyle="-",
                markersize=6,
            )
        )
        labels.append(label)
    behaviours_seen = sorted({p["behaviour"] for p in points})
    for beh in behaviours_seen:
        marker = _BEH_MARKERS.get(beh, "o")
        handles.append(
            Line2D([0], [0], color="black", marker=marker, linestyle="None", markersize=6)
        )
        labels.append(beh)

    ax.legend(handles, labels, fontsize=7, loc="best", title="id_status / behaviour")
    ax.set_xlabel("Per-arm natural-bound width  $r_a - l_a$")
    ax.set_ylabel(r"Per-arm gap to best  $\mu^* - \hat{\mu}_a$")
    ax.set_title("Bound width vs per-arm sub-optimality")
    fig.tight_layout()
    fig.savefig(output_dir / "bound_width_panel.pdf", bbox_inches="tight")
    plt.close(fig)
