"""v12 acceptance for the figure-side fixes.

* ``regime_map`` prefers ``coverage_ess_histogram`` over ``ess_ratio``
  for the y-threshold.  When both columns are present, masked runs
  with low histogram-ess but high log-prob-ess (the PR 17 bug
  signature) should land in the *Coverage-broken* quadrant.
* ``regime_map`` falls back to ``ess_ratio`` when the histogram column
  is missing (older results, no v12 columns).
* ``coverage_signature`` renders Panel C when the histogram columns
  are present, with a clear "not collected" message when they aren't.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from causal_rl.plotting.coverage_signature import make_coverage_signature  # noqa: E402
from causal_rl.plotting.regime_map import (  # noqa: E402
    make_regime_map,
    regime_quadrant_counts,
)


_HEADER_FULL = [
    "step",
    "cell",
    "env_name",
    "algorithm",
    "behaviour_policy",
    "seed",
    "eval_return_mean",
    "eval_oracle_return_mean",
    "delta_tv",
    "ess_ratio",
    "min_propensity",
    "coverage_ess_histogram",
    "coverage_action_freq_min",
    "id_status",
]
_HEADER_LEGACY = [
    "step",
    "cell",
    "env_name",
    "algorithm",
    "behaviour_policy",
    "seed",
    "eval_return_mean",
    "eval_oracle_return_mean",
    "delta_tv",
    "ess_ratio",
    "min_propensity",
    "id_status",
]


def _write_run(
    run_dir: Path,
    *,
    delta_tv: float,
    gap: float,
    ess: float,
    min_prop: float,
    cov_ess: float | None,
    cov_freq: float | None,
    id_status: str = "id",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    oracle = 18.0
    eval_ret = oracle - gap
    row = {
        "step": 100,
        "cell": 1,
        "env_name": "tabular-sepsis-v0",
        "algorithm": "dqn",
        "behaviour_policy": "uniform",
        "seed": 0,
        "eval_return_mean": eval_ret,
        "eval_oracle_return_mean": oracle,
        "delta_tv": delta_tv,
        "ess_ratio": ess,
        "min_propensity": min_prop,
        "id_status": id_status,
    }
    if cov_ess is not None:
        row["coverage_ess_histogram"] = cov_ess
        row["coverage_action_freq_min"] = cov_freq
        header = _HEADER_FULL
    else:
        header = _HEADER_LEGACY
    with (run_dir / "eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerow(row)


def test_regime_map_prefers_histogram_when_present(tmp_path: Path) -> None:
    """Masked runs should land in *Coverage-broken*, not *Identifiable*.

    The synthetic data here is the canonical PR 17 bug signature: low
    Δ_TV, high gap, but log-prob ``ess_ratio`` reads as 1.0 (the
    resampling-uniform artefact) while ``coverage_ess_histogram`` reads
    as 0.45 (truthful).  Pre-v12 these runs landed in *Identifiable*.
    """
    results = tmp_path / "results"
    # Healthy runs (anchor the y-threshold and the x-median).
    for i in range(5):
        _write_run(
            results / f"healthy_{i}",
            delta_tv=0.025,
            gap=2.0,
            ess=0.95,
            min_prop=0.10,
            cov_ess=0.95,
            cov_freq=0.10,
            id_status="id",
        )
    # Masked runs: low Δ_TV, high gap, log-prob ess looks healthy,
    # histogram ess is broken — the canonical PR 17 bug signature.
    for i in range(3):
        _write_run(
            results / f"masked_{i}",
            delta_tv=0.018,  # below x-threshold (median of id_dtv)
            gap=11.0,        # above y-threshold
            ess=1.0,         # log-prob misreports as healthy
            min_prop=0.25,
            cov_ess=0.45,    # truthful: broken
            cov_freq=0.0,
            id_status="id",
        )
    counts = regime_quadrant_counts(results)
    assert counts.get("Coverage-broken", 0) >= 3, (
        f"masked runs should land in Coverage-broken; counts={counts}"
    )
    assert counts.get("Identifiable", 0) >= 5

    # And the figure renders.
    out = tmp_path / "fig"
    make_regime_map(results, out)
    assert (out / "regime_map.pdf").exists()


def test_regime_map_falls_back_to_ess_ratio_for_legacy_runs(tmp_path: Path) -> None:
    """Pre-v12 result directories don't have the histogram column; the
    figure should still render and use ``ess_ratio`` as the fallback
    (this is the documented contract in the v12 brief)."""
    results = tmp_path / "results"
    for i in range(5):
        _write_run(
            results / f"legacy_{i}",
            delta_tv=0.02,
            gap=2.0,
            ess=0.95,
            min_prop=0.10,
            cov_ess=None,
            cov_freq=None,
        )
    counts = regime_quadrant_counts(results)
    # All legacy runs should land in *Identifiable* on the fallback.
    assert counts.get("Identifiable", 0) >= 5

    out = tmp_path / "fig"
    make_regime_map(results, out)
    assert (out / "regime_map.pdf").exists()


def test_coverage_signature_panel_c_populated_when_columns_present(
    tmp_path: Path,
) -> None:
    """With histogram columns present, ``coverage_signature`` renders
    panel C; without them, the panel shows a "not collected" message
    instead of crashing.  The smoke test in
    ``test_v11_diagnosis_figures.py`` already exercises the
    no-histogram path; this test exercises the present-histogram path.
    """
    results = tmp_path / "results"
    _write_run(
        results / "healthy",
        delta_tv=0.02,
        gap=2.0,
        ess=0.95,
        min_prop=0.10,
        cov_ess=0.95,
        cov_freq=0.10,
    )
    _write_run(
        results / "masked",
        delta_tv=0.02,
        gap=10.0,
        ess=1.0,
        min_prop=0.25,
        cov_ess=0.45,
        cov_freq=0.0,
    )
    out = tmp_path / "fig"
    make_coverage_signature(results, out)
    assert (out / "coverage_signature.pdf").exists()
