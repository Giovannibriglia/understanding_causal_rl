from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from causal_rl.plotting.style import apply_style


def _collect_rows(results_dir: Path, csv_name: str) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for csv_path in results_dir.rglob(csv_name):
        run_dir = csv_path.parent
        meta_path = run_dir / "meta.json"
        run_horizon = 20
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                run_horizon = int(meta.get("horizon", run_horizon))
            except Exception:
                run_horizon = 20
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_run_horizon"] = run_horizon
                row["_run_id"] = run_dir.name
                rows.append(row)
    return rows


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return cast(np.ndarray, y.copy())
    csum = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(y.size):
        left = max(0, i - window + 1)
        n = i - left + 1
        out[i] = (csum[i + 1] - csum[left]) / float(n)
    return out


def _build_algo_series(
    rows: list[dict[str, str | int]], value_col: str
) -> dict[tuple[int, str, str], list[tuple[np.ndarray, np.ndarray]]]:
    by_run: dict[tuple[int, str, str, str], list[tuple[int, float, int]]] = defaultdict(list)
    for r in rows:
        try:
            cell = int(r["cell"])
            algo = str(r["algorithm"])
            behaviour = str(r.get("behaviour_policy", ""))
            run_id = str(r["_run_id"])
            step = int(r["step"])
            value = float(r[value_col])
            horizon = int(r["_run_horizon"])
        except (ValueError, KeyError):
            continue
        by_run[(cell, algo, behaviour, run_id)].append((step, value, horizon))

    by_algo: dict[tuple[int, str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for (cell, algo, behaviour, _run_id), series in by_run.items():
        series_sorted = sorted(series, key=lambda x: x[0])
        xs = np.array([v[0] for v in series_sorted], dtype=np.int64)
        ys = np.array([v[1] for v in series_sorted], dtype=np.float64)
        horizon = max(1, int(series_sorted[0][2]))
        ys = _rolling_mean(ys, window=horizon)
        by_algo[(cell, algo, behaviour)].append((xs, ys))
    return by_algo


def _plot_panel(
    ax: Axes,
    cell: int,
    by_algo: dict[tuple[int, str, str], list[tuple[np.ndarray, np.ndarray]]],
    ylabel: str,
) -> None:
    from causal_rl.plotting.colors import get_style

    for (c, algo, behaviour), run_series in sorted(by_algo.items(), key=lambda x: (x[0][1], x[0][2])):
        if c != cell or not run_series:
            continue
        step_sets = [set(x.tolist()) for x, _ in run_series]
        common_steps = sorted(set.intersection(*step_sets))
        if not common_steps:
            continue
        x = np.array(common_steps, dtype=np.float64)
        stacked: list[np.ndarray] = []
        for xs, ys in run_series:
            idx = {int(step): i for i, step in enumerate(xs.tolist())}
            stacked.append(np.array([ys[idx[s]] for s in common_steps], dtype=np.float64))
        mat = np.vstack(stacked)
        median = np.median(mat, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        label = f"{algo}" + (f" ({behaviour})" if behaviour else "")
        style = get_style(algo, behaviour)
        (line,) = ax.plot(x, median, label=label, color=style.get("color"), linestyle=style.get("linestyle"))
        ax.fill_between(x, q25, q75, alpha=0.2, color=line.get_color())
    ax.set_ylabel(ylabel)
    if ax.lines:
        ax.legend(loc="best", frameon=False, fontsize=7)


def _oracle_curve(
    rows: list[dict[str, str | int]], cell: int
) -> tuple[np.ndarray, np.ndarray] | None:
    points: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        try:
            if int(r["cell"]) != cell:
                continue
            step = int(r["step"])
            value = float(r["eval_oracle_return_mean"])
        except (KeyError, ValueError):
            continue
        points[step].append(value)
    if not points:
        return None
    steps = np.array(sorted(points.keys()), dtype=np.float64)
    vals = np.array([np.median(points[int(s)]) for s in steps], dtype=np.float64)
    return steps, vals


def make_learning_curves(results_dir: Path, output_dir: Path, env_prefix: str | None = None) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _collect_rows(results_dir, "train.csv")
    eval_rows = _collect_rows(results_dir, "eval.csv")
    if env_prefix is not None:
        train_rows = [r for r in train_rows if str(r.get("env_name", "")).startswith(env_prefix)]
        eval_rows = [r for r in eval_rows if str(r.get("env_name", "")).startswith(env_prefix)]
    by_algo_train = _build_algo_series(train_rows, "train_return_mean")
    by_algo_eval = _build_algo_series(eval_rows, "eval_return_mean")

    for cell in range(1, 9):
        fig, axes = plt.subplots(2, 1, figsize=(6.75, 4.8), sharex=True)
        _plot_panel(axes[0], cell, by_algo_train, ylabel="Train Return")
        _plot_panel(axes[1], cell, by_algo_eval, ylabel="Eval Return")
        oracle = _oracle_curve(eval_rows, cell)
        if oracle is not None:
            ox, oy = oracle
            axes[1].plot(ox, oy, color="black", linestyle="--", linewidth=1.2, label="oracle")
            axes[1].legend(loc="best", frameon=False, fontsize=7)
        axes[1].set_xlabel("Step")
        fig.suptitle(f"Cell {cell}")
        pdf = output_dir / f"learning_curves_cell{cell}.pdf"
        png = output_dir / f"learning_curves_cell{cell}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
