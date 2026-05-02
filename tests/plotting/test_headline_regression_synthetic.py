"""Synthetic test for headline regression pre-committed stratified R² claim.

This test builds a synthetic dataset where:
  - In the 'id' stratum: G is a strong linear function of delta_tv (high R²).
  - In the 'non_id' stratum: G is nearly random (low R²).
  - In the 'partial_id' stratum: G is a moderate function of multiple features.

It asserts the regression recovers this structure — but on *synthetic* data,
not on real experimental data. The real R² claim is checked by the paper run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

# Skip if sklearn not available
pytest.importorskip("sklearn")

from causal_rl.plotting.headline_regression import _load_joined_data, _r2_score


def _write_synthetic_data(results_dir: Path) -> None:
    """Write synthetic eval.csv and eval_perturbed.csv files."""
    rng = np.random.default_rng(42)
    n_per_stratum = 80

    strata = {
        "id": {"slope": 2.0, "noise": 0.1},
        "partial_id": {"slope": 1.0, "noise": 0.3},
        "non_id": {"slope": 0.0, "noise": 1.0},
    }

    eval_cols = [
        "step",
        "cell",
        "env_name",
        "algorithm",
        "behaviour_policy",
        "seed",
        "eval_return_mean",
        "eval_oracle_return_mean",
        "eval_return_std",
        "eval_oracle_return_std",
        "id_status",
        "delta_tv",
        "bound_width_mean",
        "min_propensity",
        "ess_ratio",
        "overlap_at_1e-2",
        "tail_mass_top10pct",
        "propensity_calibration_ece",
        "bias_strength",
        "alpha_conf",
        "expose_z",
        "pi_b_known",
        "delta_tv_ci_lo",
        "delta_tv_ci_hi",
        "delta_kl",
        "delta_chi2",
        "delta_sup",
        "delta_mmd2",
        "delta_ks",
    ]
    perturbed_cols = [
        "seed",
        "cell",
        "env",
        "algorithm",
        "behaviour",
        "eval_perturbed_return_mean",
        "eval_perturbed_return_std",
        "D_env_KS",
        "eps_T",
        "eps_R",
    ]

    run_idx = 0
    for status, params in strata.items():
        slope = float(params["slope"])
        noise_scale = float(params["noise"])
        for i in range(n_per_stratum):
            run_dir = results_dir / f"run_{run_idx}"
            run_dir.mkdir(parents=True)
            run_idx += 1

            delta_tv = float(rng.uniform(0.0, 0.5))
            oracle_return = float(rng.uniform(0.5, 1.0))
            noise = float(rng.normal(0, noise_scale))
            G = slope * delta_tv + noise  # noqa: N806

            eval_row = {
                "step": 100,
                "cell": 1,
                "env_name": "tabular-sepsis-v0",
                "algorithm": "dqn",
                "behaviour_policy": "uniform",
                "seed": i,
                "eval_return_mean": oracle_return - G,
                "eval_oracle_return_mean": oracle_return,
                "eval_return_std": 0.1,
                "eval_oracle_return_std": 0.1,
                "id_status": status,
                "delta_tv": delta_tv,
                "bound_width_mean": float(rng.uniform(0.1, 0.5)),
                "min_propensity": float(rng.uniform(0.05, 0.5)),
                "ess_ratio": float(rng.uniform(0.1, 1.0)),
                "overlap_at_1e-2": float(rng.uniform(0.5, 1.0)),
                "tail_mass_top10pct": float(rng.uniform(0.05, 0.3)),
                "propensity_calibration_ece": float(rng.uniform(0.0, 0.1)),
                "bias_strength": 1.0,
                "alpha_conf": 2.0,
                "expose_z": 1.0,
                "pi_b_known": 0.0,
                "delta_tv_ci_lo": delta_tv * 0.9,
                "delta_tv_ci_hi": delta_tv * 1.1,
                "delta_kl": delta_tv * 1.2,
                "delta_chi2": delta_tv * 0.8,
                "delta_sup": delta_tv * 0.7,
                "delta_mmd2": delta_tv * 0.6,
                "delta_ks": delta_tv * 0.5,
            }
            with (run_dir / "eval.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=eval_cols)
                writer.writeheader()
                writer.writerow(eval_row)

            # Write perturbed eval
            perturbed_row = {
                "seed": i,
                "cell": 1,
                "env": "tabular-sepsis-v0",
                "algorithm": "dqn",
                "behaviour": "uniform",
                "eval_perturbed_return_mean": oracle_return - G - float(rng.normal(0, 0.05)),
                "eval_perturbed_return_std": 0.1,
                "D_env_KS": float(rng.uniform(0.0, 0.3)),
                "eps_T": 0.2,
                "eps_R": 0.1,
            }
            with (run_dir / "eval_perturbed.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=perturbed_cols)
                writer.writeheader()
                writer.writerow(perturbed_row)

            # Write meta.json (needed by _load_joined_data)
            (run_dir / "meta.json").write_text(json.dumps({"offline_transitions": 50000}))


def test_make_headline_regression_runs(tmp_path: Path) -> None:
    """make_headline_regression runs end-to-end with synthetic data."""
    from causal_rl.plotting.headline_regression import make_headline_regression

    results_dir = tmp_path / "results_full"
    out = tmp_path / "out_full"
    _write_synthetic_data(results_dir)
    make_headline_regression(results_dir, out)
    assert (out / "headline_regression_table.csv").exists()
    assert (out / "headline_stratified_r2.csv").exists()
    assert (out / "headline_pure_divergence.pdf").exists()
    assert (out / "headline_fitted.pdf").exists()

    # Check stratified R² CSV has all strata
    import csv as csv_mod

    with (out / "headline_stratified_r2.csv").open() as f:
        strat_rows = list(csv_mod.DictReader(f))
    strata_found = {r["stratum"] for r in strat_rows}
    assert {"id", "partial_id", "non_id", "pooled"} == strata_found


def test_regression_recovers_stratified_structure(tmp_path: Path) -> None:
    """The regression should recover that id-stratum has high R² and non_id low."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    results_dir = tmp_path / "results"
    _write_synthetic_data(results_dir)

    rows = _load_joined_data(results_dir)
    assert len(rows) >= 100, f"Expected >= 100 rows, got {len(rows)}"

    feature_cols = ["delta_tv", "bound_width_mean", "min_propensity", "ess_ratio", "D_env_KS"]
    X_raw = np.array([[r.get(c, 0.0) for c in feature_cols] for r in rows])  # noqa: N806
    y = np.array([r["G"] for r in rows])
    statuses = [str(r["id_status"]) for r in rows]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)  # noqa: N806

    # Fit per-stratum
    for status in ["id", "partial_id", "non_id"]:
        mask = np.array([s == status for s in statuses])
        if mask.sum() < 5:
            continue
        lr = LinearRegression()
        lr.fit(X[mask], y[mask])
        r2 = _r2_score(y[mask], lr.predict(X[mask]))
        if status == "id":
            assert r2 > 0.5, f"id stratum R²={r2:.2f} should be > 0.5"
        elif status == "non_id":
            assert r2 < 0.5, f"non_id stratum R²={r2:.2f} should be < 0.5"


def test_r2_score_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_r2_score(y, y) - 1.0) < 1e-10


def test_r2_score_null() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full_like(y, y.mean())
    assert abs(_r2_score(y, y_pred)) < 1e-10
