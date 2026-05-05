"""v15 contract: ``summary.json`` and ``headline_regression_meta.json``
agree on every R² value.

Pre-v15, ``scripts/make_summary.py`` fitted its own ``LinearRegression``
on the joined eval+perturbed data while
``causal_rl.plotting.headline_regression.make_headline_regression``
fitted ``RidgeCV`` on the same X / y.  The two reported different
``r2_pooled_cv`` values for the same data — 0.484 (OLS) vs 0.957
(Ridge), with the OLS path producing catastrophic per-stratum CV R²
inside ``partial_id`` due to overfitting.

The v15 fix has ``make_summary._compute_regression`` delegate to
``make_headline_regression``: writes the JSON / CSV outputs into the
results dir, then reads them.  Single source of truth → impossible
for the two outputs to drift.

This test exercises the contract: build a synthetic results tree,
run ``make_summary``, then compare the regression block against the
``headline_regression_meta.json`` / ``headline_stratified_r2.csv`` that
``make_headline_regression`` wrote into the same directory.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


_EVAL_HEADER = [
    "step",
    "cell",
    "env_name",
    "algorithm",
    "behaviour_policy",
    "seed",
    "eval_return_mean",
    "eval_oracle_return_mean",
    "delta_tv",
    "delta_tv_ci_lo",
    "delta_tv_ci_hi",
    "delta_kl",
    "delta_chi2",
    "delta_sup",
    "delta_mmd2",
    "delta_ks",
    "min_propensity",
    "ess_ratio",
    "overlap_at_1e-2",
    "tail_mass_top10pct",
    "propensity_calibration_ece",
    "pi_b_recovery_kl",
    "cond_mi_r_z_given_sa",
    "target_support_overlap",
    "backdoor_residual_mean",
    "bound_width_mean",
    "bound_width_max",
    "bound_width_std",
    "bias_strength",
    "alpha_conf",
    "expose_z",
    "pi_b_known",
    "id_status",
]


def _write_run(
    run_dir: Path,
    *,
    cell: int,
    seed: int,
    delta_tv: float,
    eval_ret: float,
    oracle_ret: float,
    id_status: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "step": 100,
        "cell": cell,
        "env_name": "tabular-sepsis-v0",
        "algorithm": "dqn",
        "behaviour_policy": "reward_aligned",
        "seed": seed,
        "eval_return_mean": eval_ret,
        "eval_oracle_return_mean": oracle_ret,
        "delta_tv": delta_tv,
        "delta_tv_ci_lo": delta_tv * 0.9,
        "delta_tv_ci_hi": delta_tv * 1.1,
        "delta_kl": delta_tv * 1.2,
        "delta_chi2": delta_tv * 0.9,
        "delta_sup": delta_tv * 0.7,
        "delta_mmd2": delta_tv * 0.6,
        "delta_ks": delta_tv * 0.5,
        "min_propensity": 0.10,
        "ess_ratio": 0.95,
        "overlap_at_1e-2": 0.85,
        "tail_mass_top10pct": 0.10,
        "propensity_calibration_ece": 0.05,
        "pi_b_recovery_kl": 0.02,
        "cond_mi_r_z_given_sa": 0.1 if cell <= 2 else 0.4,
        "target_support_overlap": 0.85,
        "backdoor_residual_mean": 0.05 if cell <= 2 else float("nan"),
        "bound_width_mean": 0.875,
        "bound_width_max": 0.875 + 0.02 * cell,
        "bound_width_std": 0.01 * cell,
        "bias_strength": 1.0,
        "alpha_conf": 2.0,
        "expose_z": 1.0 if cell <= 2 else 0.0,
        "pi_b_known": 1.0 if cell in (1, 3) else 0.0,
        "id_status": id_status,
    }
    with (run_dir / "eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_EVAL_HEADER)
        w.writeheader()
        w.writerow(row)


def _build_synthetic_results(results_dir: Path) -> None:
    """Build a small tree spanning all three id_status strata so the
    regression has enough data to fit (need ≥10 rows in
    ``make_headline_regression``)."""
    n_per_status = 12
    statuses = (
        ("id", 1, 2.0),
        ("partial_id", 3, 6.0),
        ("non_id", 4, 12.0),
    )
    idx = 0
    for status, cell, target_gap in statuses:
        for seed in range(n_per_status):
            # Vary delta_tv slightly so the regression has signal.
            dtv = 0.04 + 0.005 * (seed % 4)
            oracle = 18.0
            # Inject a gap that loosely tracks delta_tv so OLS / Ridge
            # both find a non-trivial relationship.
            gap = target_gap + 0.5 * (seed % 3)
            _write_run(
                results_dir / f"run_{idx}",
                cell=cell,
                seed=seed,
                delta_tv=dtv,
                eval_ret=oracle - gap,
                oracle_ret=oracle,
                id_status=status,
            )
            idx += 1


def _approx_eq(a: object, b: object, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        af = float(a)  # type: ignore[arg-type]
        bf = float(b)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if math.isnan(af) and math.isnan(bf):
        return True
    return abs(af - bf) <= tol


def test_summary_regression_matches_headline_regression(tmp_path: Path) -> None:
    """``summary.json`` must match ``headline_regression_meta.json`` /
    ``headline_stratified_r2.csv`` exactly on every R² field."""
    from make_summary import make_summary

    results = tmp_path / "results"
    results.mkdir(parents=True)
    _build_synthetic_results(results)

    summary_path = make_summary(results, output_path=results / "summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reg = summary["headline_regression"]
    # Skip if the regression returned an error (e.g. sklearn missing in CI).
    assert "r2_pooled_cv" in reg, (
        f"regression block missing r2_pooled_cv; got {reg}"
    )

    # Read the source-of-truth files written by make_headline_regression.
    meta_path = results / "headline_regression_meta.json"
    strat_path = results / "headline_stratified_r2.csv"
    assert meta_path.exists(), "headline_regression_meta.json was not written"
    assert strat_path.exists(), "headline_stratified_r2.csv was not written"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with strat_path.open("r", encoding="utf-8") as f:
        strat_rows = {row["stratum"]: row for row in csv.DictReader(f)}

    # Pooled R² values come from the meta JSON.
    assert _approx_eq(reg["r2_pooled_train"], meta["r2_pooled_train"]), (
        f"pooled train R²: summary={reg['r2_pooled_train']} vs "
        f"meta={meta['r2_pooled_train']}"
    )
    assert _approx_eq(reg["r2_pooled_cv"], meta["r2_pooled_cv"]), (
        f"pooled CV R²: summary={reg['r2_pooled_cv']} vs "
        f"meta={meta['r2_pooled_cv']}"
    )

    # Per-stratum R² values come from the stratified CSV.
    for stratum in ("id", "partial_id", "non_id"):
        if stratum not in strat_rows:
            continue
        strat = strat_rows[stratum]
        for kind in ("r2_train", "r2_cv"):
            csv_val = float(strat[kind]) if strat[kind] not in ("", "nan") else None
            summary_key = f"r2_{stratum}_{kind.split('_')[1]}"
            summary_val = reg.get(summary_key)
            assert _approx_eq(summary_val, csv_val), (
                f"{summary_key}: summary={summary_val} vs csv={csv_val}"
            )

    # The new metadata fields must be present.
    assert reg.get("estimator") == "RidgeCV", (
        f"v15 contract: estimator should be RidgeCV, got {reg.get('estimator')!r}"
    )
    assert reg.get("mode") in {"in_domain", "generalisation"}, (
        f"unexpected mode {reg.get('mode')!r}"
    )
