from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style, figure


def _load_eval(results_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in results_dir.rglob("eval.csv"):
        with p.open("r", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def make_ablations(results_dir: Path, output_dir: Path, env_prefix: str | None = None) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_eval(results_dir)
    if env_prefix is not None:
        rows = [r for r in rows if str(r.get("env_name", "")).startswith(env_prefix)]
    grouped: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        try:
            grouped[int(r["cell"])].append(float(r["delta_tv"]))
        except (ValueError, KeyError):
            continue
    xs = np.array(sorted(grouped.keys()), dtype=np.float64)
    ys = (
        np.array([np.mean(grouped[k]) for k in sorted(grouped.keys())], dtype=np.float64)
        if len(xs)
        else np.array([])
    )

    fig, ax = figure()
    if len(xs):
        ax.plot(xs, ys, marker="o")
    ax.set_xlabel(r"$\alpha_{\mathrm{conf}}$ proxy (cell index)")
    ax.set_ylabel(r"$\widehat{\Delta}_{\mathrm{TV}}$")
    fig.savefig(output_dir / "ablations_alpha.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ablations_alpha.png", dpi=500, bbox_inches="tight")
    plt.close(fig)

    fig, ax = figure()
    if len(xs):
        ax.plot(xs, 1.0 / (1.0 + ys + 1e-6), marker="o")
    ax.set_xlabel("Coverage knob")
    ax.set_ylabel("Coverage score")
    fig.savefig(output_dir / "ablations_coverage.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ablations_coverage.png", dpi=500, bbox_inches="tight")
    plt.close(fig)

    fig, ax = figure()
    if len(xs):
        ax.bar(xs, ys)
    ax.set_xlabel("Observability fraction")
    ax.set_ylabel(r"$\widehat{\Delta}_{\mathrm{TV}}$")
    fig.savefig(output_dir / "ablations_observability.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ablations_observability.png", dpi=500, bbox_inches="tight")
    plt.close(fig)
