from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

DIVERGENCE_COLS = [
    ("delta_tv", r"$\widehat{\Delta}_{\mathrm{TV}}$"),
    ("delta_kl", r"$\widehat{\Delta}_{\mathrm{KL}}$"),
    ("delta_chi2", r"$\widehat{\Delta}_{\chi^2}$"),
    ("delta_sup", r"$\widehat{\Delta}_{\sup}$"),
]


def make_gap_curves(results_dir: Path, output_dir: Path, env_prefix: str | None = None) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for path in results_dir.rglob("train.csv"):
        with path.open("r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    if env_prefix is not None:
        rows = [r for r in rows if str(r.get("env_name", "")).startswith(env_prefix)]
    by_series: dict[tuple[int, str, str, str, int], dict[str, list[tuple[int, float]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        try:
            key = (
                int(r["cell"]),
                str(r["algorithm"]),
                str(r.get("behaviour_policy", "unknown")),
                str(r.get("env_name", "env")),
                int(r.get("seed", 0)),
            )
            step = int(r["step"])
        except (ValueError, KeyError):
            continue
        for col, _ in DIVERGENCE_COLS:
            try:
                val = float(r[col])
            except (ValueError, KeyError):
                continue
            by_series[key][col].append((step, val))

    by_cell_label: dict[tuple[int, str], dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for (cell, algo, beh, env_name, seed), metric_map in by_series.items():
        label = f"{algo}/{beh}/{env_name.split('-')[0]}"
        for col, values in metric_map.items():
            vals_sorted = sorted(values, key=lambda x: x[0])
            xs = np.array([v[0] for v in vals_sorted], dtype=np.int64)
            ys = np.array([v[1] for v in vals_sorted], dtype=np.float64)
            by_cell_label[(cell, label)][col][seed] = (xs, ys)

    from causal_rl.plotting.colors import get_style, parse_algo_beh_from_label

    for cell in range(1, 9):
        fig, axes = plt.subplots(2, 2, figsize=(6.75, 4.8), sharex=True)
        axes_flat = axes.flatten()
        cell_data = {k[1]: v for k, v in by_cell_label.items() if k[0] == cell}
        for ax, (col, ylabel) in zip(axes_flat, DIVERGENCE_COLS, strict=True):
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
                    stacked.append(np.array([ys[idx[s]] for s in common_steps], dtype=np.float64))
                mat = np.vstack(stacked)
                median = np.median(mat, axis=0)
                q25 = np.percentile(mat, 25, axis=0)
                q75 = np.percentile(mat, 75, axis=0)
                algo, beh = parse_algo_beh_from_label(label)
                style = get_style(algo, beh)
                (line,) = ax.plot(
                    x,
                    median,
                    label=label,
                    color=style.get("color"),
                    linestyle=style.get("linestyle"),
                )
                ax.fill_between(x, q25, q75, alpha=0.2, color=line.get_color())
            ax.set_ylabel(ylabel)
            if ax.lines:
                ax.legend(loc="best", frameon=False, fontsize=6)
        for ax in axes[1, :]:
            ax.set_xlabel("Step")
        fig.suptitle(f"Cell {cell}")
        pdf = output_dir / f"gap_curves_cell{cell}.pdf"
        png = output_dir / f"gap_curves_cell{cell}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
