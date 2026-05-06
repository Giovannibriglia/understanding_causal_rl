"""Bias / coverage sweep figure.

Four subplots sharing the x-axis (``bias_strength``):
  Panel 0: Δ̂_TV vs bias_strength (one line per id_status).
  Panel 1: ESS vs bias_strength on log-y.
  Panel 2: ``min_propensity`` vs bias_strength on log-y.
  Panel 3: return gap to oracle vs bias_strength.

Lines are coloured by ``id_status`` (the paper's primary stratification
axis), with shaded bootstrap CIs from ``delta_tv_ci_lo / ci_hi``.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.plotting.style import apply_style

_ID_STATUS_COLORS = {"id": "#4CAF50", "partial_id": "#FF9800", "non_id": "#F44336"}


def make_bias_sweep(
    results_dir: Path,
    output_dir: Path,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data: {id_status: {bias_strength: {metric: [values]}}}
    data: dict[str, dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for eval_path in results_dir.rglob("eval.csv"):
        meta_path = eval_path.parent / "meta.json"
        if not meta_path.exists():
            continue
        with eval_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = rows[-1]
        # v15: drop uniform-behaviour runs.  ``UniformExplorer`` has no
        # ``bias_strength`` parameter and reports 1.0 for every run
        # regardless of the matrix sweep value, so all uniform runs
        # land in the bs=1.0 bucket — they pull that bucket's mean
        # toward "uniform-like" (high ESS, high min_propensity) and
        # produce a visible jump-back at bs=1.0 in the id / partial_id
        # strata.  Uniform doesn't have a meaningful bias_strength axis
        # in the first place; plotting it against a swept bs is a
        # category error.  See ``docs/v15_bias_sweep_investigation.md``.
        if str(last.get("behaviour_policy", "")) == "uniform":
            continue
        try:
            bias_strength = float(last.get("bias_strength", 1.0))
            id_status = str(last.get("id_status", "non_id"))
            delta_tv = float(last.get("delta_tv", 0.0))
            ci_lo = float(last.get("delta_tv_ci_lo", delta_tv))
            ci_hi = float(last.get("delta_tv_ci_hi", delta_tv))
            gap = float(last.get("eval_oracle_return_mean", 0.0)) - float(
                last.get("eval_return_mean", 0.0)
            )
            min_prop = float(last.get("min_propensity", 0.0))
            ess = float(last.get("ess_ratio", 0.0))
        except (ValueError, KeyError):
            continue
        d = data[id_status][bias_strength]
        d["delta_tv"].append(delta_tv)
        d["ci_lo"].append(ci_lo)
        d["ci_hi"].append(ci_hi)
        d["gap"].append(gap)
        d["min_propensity"].append(min_prop)
        d["ess_ratio"].append(ess)

    if not data:
        return

    # v16: detect the degenerate single-bias_strength case and render an
    # honest figure.  The line-plot path below asserts trends in its
    # titles ("Δ_TV rises with bias_strength", etc.); when only one
    # bias_strength value is present in the manifest those trends are
    # not depicted, so the line-plot output ends up misleading.  The
    # single-bs branch swaps the lines for stratified bar charts at the
    # one bias_strength, drops the trend framing, and adds a banner
    # noting that a sweep is required to recover the trend figure.
    all_bs = sorted({bs for by_bs in data.values() for bs in by_bs.keys()})
    single_bs = len(all_bs) == 1

    if single_bs:
        bs_value = all_bs[0]
        statuses_present = [s for s in ("id", "partial_id", "non_id") if s in data]
        x_positions = np.arange(len(statuses_present))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=False)
        ax_tv, ax_ess, ax_prop, ax_gap = axes

        def _means(metric: str) -> list[float]:
            out: list[float] = []
            for status in statuses_present:
                vals = data[status].get(bs_value, {}).get(metric, [])
                out.append(float(np.mean(vals)) if vals else 0.0)
            return out

        colors = [_ID_STATUS_COLORS.get(s, "gray") for s in statuses_present]
        for ax, metric, ylabel in (
            (ax_tv, "delta_tv", r"$\widehat{\Delta}_{\mathrm{TV}}$"),
            (ax_ess, "ess_ratio", "ESS / N"),
            (ax_prop, "min_propensity", "min propensity"),
            (ax_gap, "gap", "Return gap to oracle"),
        ):
            means = _means(metric)
            # ESS / min_propensity preserve their log-y scale.  Floor
            # at 1e-6 so log scale handles zero-stratum cases.
            if metric in ("ess_ratio", "min_propensity"):
                means = [max(1e-6, m) for m in means]
            ax.bar(
                x_positions,
                means,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels(statuses_present)
            ax.set_xlabel("id_status")
            ax.set_ylabel(ylabel)

        ax_ess.set_yscale("log")
        ax_prop.set_yscale("log")

        ax_tv.set_title(rf"$\widehat{{\Delta}}_{{\mathrm{{TV}}}}$ at bs={bs_value}, by id_status")
        ax_ess.set_title(f"ESS / N at bs={bs_value}, by id_status")
        ax_prop.set_title(f"min propensity at bs={bs_value}, by id_status")
        ax_gap.set_title(f"Return gap at bs={bs_value}, by id_status")

        # Single-bs banner replaces the multi-bs uniform-omission
        # caption.  Both share the same fig.text slot at y=-0.07.
        fig.suptitle(
            "Bias / coverage sweep, stratified by identifiability status",
            fontsize=11,
        )
        fig.text(
            0.5,
            -0.07,
            "Single bias_strength in this run; bias_strength trend not "
            "depicted.  Sweep bias_strengths to recover the trend figure.",
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
            style="italic",
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(output_dir / "bias_sweep.pdf", bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    ax_tv, ax_ess, ax_prop, ax_gap = axes

    for status in ["id", "partial_id", "non_id"]:
        if status not in data:
            continue
        color = _ID_STATUS_COLORS.get(status, "gray")
        bs_vals = sorted(data[status].keys())
        if not bs_vals:
            continue
        x = np.array(bs_vals, dtype=np.float64)
        tv_means = np.array([float(np.mean(data[status][b]["delta_tv"])) for b in bs_vals])
        ci_lo_m = np.array([float(np.mean(data[status][b]["ci_lo"])) for b in bs_vals])
        ci_hi_m = np.array([float(np.mean(data[status][b]["ci_hi"])) for b in bs_vals])
        gap_means = np.array([float(np.mean(data[status][b]["gap"])) for b in bs_vals])
        prop_means = np.array(
            [max(1e-6, float(np.mean(data[status][b]["min_propensity"]))) for b in bs_vals]
        )
        ess_means = np.array(
            [max(1e-6, float(np.mean(data[status][b]["ess_ratio"]))) for b in bs_vals]
        )

        ax_tv.plot(x, tv_means, color=color, marker="o", label=status)
        ax_tv.fill_between(x, ci_lo_m, ci_hi_m, color=color, alpha=0.15)

        ax_ess.plot(x, ess_means, color=color, marker="o", label=status)
        ax_prop.plot(x, prop_means, color=color, marker="o", label=status)
        ax_gap.plot(x, gap_means, color=color, marker="o", label=status)

    ax_tv.set_xlabel("bias_strength")
    ax_tv.set_ylabel(r"$\widehat{\Delta}_{\mathrm{TV}}$")
    ax_tv.set_title(r"$\widehat{\Delta}_{\mathrm{TV}}$ rises with bias_strength")

    ax_ess.set_xlabel("bias_strength")
    ax_ess.set_ylabel("ESS / N")
    ax_ess.set_yscale("log")
    ax_ess.set_title("ESS falls with bias_strength")

    ax_prop.set_xlabel("bias_strength")
    ax_prop.set_ylabel("min propensity")
    ax_prop.set_yscale("log")
    ax_prop.set_title("min propensity falls with bias_strength")

    ax_gap.set_xlabel("bias_strength")
    ax_gap.set_ylabel("Return gap to oracle")
    ax_gap.set_title("Return gap rises with bias_strength")

    handles, labels = ax_tv.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="id_status",
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
            fontsize=9,
        )
    fig.suptitle("Bias / coverage sweep, stratified by identifiability status", fontsize=11)
    # v15: caption note explaining why uniform runs are omitted.  See
    # docs/v15_bias_sweep_investigation.md for the full rationale.
    fig.text(
        0.5,
        -0.07,
        "Uniform-behaviour runs are omitted: bias_strength is not "
        "meaningful under a uniform behaviour policy.",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
        style="italic",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output_dir / "bias_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
