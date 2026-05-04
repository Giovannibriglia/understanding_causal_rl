"""Per-cell time-series of *diagnostic* metrics over training.

Counterpart to ``gap_curves`` which plots the divergence family
(delta_tv, delta_kl, delta_chi2, delta_sup) per cell over training step.
This module plots the *coverage* and *causal-RL* diagnostics — the
metrics that exist in every CSV but had zero plot references prior to
v11.

Six subplots per cell:

  - min_propensity
  - ess_ratio
  - propensity_calibration_ece
  - pi_b_recovery_kl
  - cond_mi_r_z_given_sa
  - bound_width_mean

One line per (algorithm, behaviour); shaded band is the seed-stratified
inter-quartile range.  When a metric is missing from the CSV (e.g.
``cond_mi_r_z_given_sa`` is NaN in cells without exposed Z), the
corresponding subplot shows a "not collected" annotation rather than
silently rendering a flat zero line.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

DIAGNOSTIC_COLS: tuple[tuple[str, str], ...] = (
    ("min_propensity", "min_propensity"),
    ("ess_ratio", "ESS / N"),
    ("propensity_calibration_ece", "ECE (π_b model)"),
    ("pi_b_recovery_kl", "KL(true ‖ pred π_b)"),
    ("cond_mi_r_z_given_sa", r"$I(R; Z \mid S, A)$"),
    ("bound_width_mean", "mean bound width"),
)


def _safe_float(val: object) -> float:
    if val is None or val == "" or val == "nan":
        return float("nan")
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def make_metrics_curves(
    results_dir: Path,
    output_dir: Path,
    env_prefix: str | None = None,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for path in results_dir.rglob("train.csv"):
        with path.open("r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    if env_prefix is not None:
        rows = [r for r in rows if str(r.get("env_name", "")).startswith(env_prefix)]
    if not rows:
        print("[metrics_curves] no train.csv rows found; skipping.")
        return

    # by_series[(cell, algo, beh, env_name, seed)][metric] = list[(step, val)]
    by_series: dict[
        tuple[int, str, str, str, int], dict[str, list[tuple[int, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            cell = int(r["cell"])
            algo = str(r["algorithm"])
            beh = str(r.get("behaviour_policy", "unknown"))
            env_name = str(r.get("env_name", "env"))
            seed = int(r.get("seed", 0))
            step = int(r["step"])
        except (ValueError, KeyError):
            continue
        key = (cell, algo, beh, env_name, seed)
        for col, _ in DIAGNOSTIC_COLS:
            v = _safe_float(r.get(col))
            if not np.isfinite(v):
                continue
            by_series[key][col].append((step, v))

    by_cell_label: dict[
        tuple[int, str], dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for (cell, algo, beh, env_name, seed), metric_map in by_series.items():
        label = f"{algo}/{beh}/{env_name.split('-')[0]}"
        for col, values in metric_map.items():
            vals_sorted = sorted(values, key=lambda x: x[0])
            xs = np.array([v[0] for v in vals_sorted], dtype=np.int64)
            ys = np.array([v[1] for v in vals_sorted], dtype=np.float64)
            by_cell_label[(cell, label)][col][seed] = (xs, ys)

    from causal_rl.plotting.colors import (
        get_color,
        get_linestyle,
        parse_algo_beh_from_label,
    )

    cells_present = sorted({k[0] for k in by_cell_label})
    for cell in cells_present:
        fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.0), sharex=True)
        axes_flat = axes.flatten()
        cell_data = {k[1]: v for k, v in by_cell_label.items() if k[0] == cell}
        for ax, (col, ylabel) in zip(axes_flat, DIAGNOSTIC_COLS, strict=True):
            any_drawn = False
            for label, metric_seed_map in cell_data.items():
                seed_series = metric_seed_map.get(col, {})
                if not seed_series:
                    continue
                step_sets = [set(x.tolist()) for x, _ in seed_series.values()]
                common_steps = sorted(set.intersection(*step_sets))
                if not common_steps:
                    continue
                x = np.array(common_steps, dtype=np.float64)
                stacked: list[np.ndarray] = []
                for xs, ys in seed_series.values():
                    idx = {int(step): i for i, step in enumerate(xs.tolist())}
                    stacked.append(
                        np.array([ys[idx[s]] for s in common_steps], dtype=np.float64)
                    )
                mat = np.vstack(stacked)
                median = np.median(mat, axis=0)
                q25 = np.percentile(mat, 25, axis=0)
                q75 = np.percentile(mat, 75, axis=0)
                algo_l, beh_l = parse_algo_beh_from_label(label)
                (line,) = ax.plot(
                    x,
                    median,
                    label=label,
                    color=get_color(algo_l, beh_l),
                    linestyle=get_linestyle(algo_l, beh_l),
                )
                ax.fill_between(x, q25, q75, alpha=0.2, color=line.get_color())
                any_drawn = True
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            if not any_drawn:
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
        for ax in axes[1, :]:
            ax.set_xlabel("Step")
        # Single legend at the bottom shared across panels.
        handles, labels = [], []
        for ax in axes_flat:
            for line, lbl in zip(ax.get_lines(), [l.get_label() for l in ax.get_lines()], strict=False):
                if lbl not in labels and not lbl.startswith("_"):
                    handles.append(line)
                    labels.append(lbl)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(4, len(handles)),
                frameon=False,
                fontsize=7,
            )
        fig.suptitle(f"Diagnostic metrics over training, cell {cell}")
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        pdf = output_dir / f"metrics_curves_cell{cell}.pdf"
        png = output_dir / f"metrics_curves_cell{cell}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
