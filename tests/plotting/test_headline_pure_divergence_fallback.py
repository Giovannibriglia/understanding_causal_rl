"""v20: ``headline_pure_divergence.pdf`` falls back to a single
divergence column when the other is dropped as a constant feature.

Pre-v20 the figure required *both* ``delta_tv`` and ``D_env_KS`` to be
in ``feature_cols``.  When ``D_env_KS`` was dropped as a constant
feature (which happens whenever ``eval_perturbations=false``, e.g.
every bandit run), the figure rendered as empty axes with the "Pure
divergence predictor" title — a misleading PDF shipped to the paper
pipeline.

v20 picks the available divergence column instead.  The fallback
logic is factored as ``_select_divergence_axis`` for direct unit
testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from causal_rl.plotting.headline_regression import _select_divergence_axis


def test_select_divergence_axis_prefers_sum_when_both_present() -> None:
    X = np.array([[0.1, 1.0, 0.5], [0.2, 1.0, 0.6], [0.3, 1.0, 0.7]])
    feature_cols = ["delta_tv", "bound_width_mean", "D_env_KS"]
    result = _select_divergence_axis(X, feature_cols)
    assert result is not None
    combined, label = result
    assert label == "Δ_TV + D_env_KS"
    np.testing.assert_array_equal(combined, X[:, 0] + X[:, 2])


def test_select_divergence_axis_falls_back_to_delta_tv() -> None:
    """When D_env_KS is missing (dropped as constant or never present),
    fall back to delta_tv alone."""
    X = np.array([[0.1, 1.0], [0.2, 1.0], [0.3, 1.0]])
    feature_cols = ["delta_tv", "bound_width_mean"]
    result = _select_divergence_axis(X, feature_cols)
    assert result is not None
    combined, label = result
    assert label == "Δ_TV", (
        f"fallback should report only the column actually plotted; got {label!r}"
    )
    assert "D_env_KS" not in label, (
        "label should not mention D_env_KS when D_env_KS is not in feature_cols"
    )
    np.testing.assert_array_equal(combined, X[:, 0])


def test_select_divergence_axis_falls_back_to_d_env_ks() -> None:
    """Symmetric fallback: only D_env_KS available."""
    X = np.array([[1.0, 0.5], [1.0, 0.6], [1.0, 0.7]])
    feature_cols = ["bound_width_mean", "D_env_KS"]
    result = _select_divergence_axis(X, feature_cols)
    assert result is not None
    combined, label = result
    assert label == "D_env_KS"
    np.testing.assert_array_equal(combined, X[:, 1])


def test_select_divergence_axis_returns_none_when_neither_present() -> None:
    """If both divergence columns are missing, the helper signals to the
    caller to skip the figure."""
    X = np.array([[1.0], [1.0], [1.0]])
    feature_cols = ["bound_width_mean"]
    assert _select_divergence_axis(X, feature_cols) is None


def test_pure_divergence_renders_when_d_env_ks_dropped(tmp_path: Path) -> None:
    """End-to-end: when ``D_env_KS`` is constant across all runs (and
    therefore dropped by the constant-feature filter), the figure
    pipeline still produces a non-empty ``headline_pure_divergence.pdf``.

    Pre-v20 the figure rendered with empty axes (no scatter, no fit
    lines) — the file existed but was visually blank because the
    ``if div_col >= 0 and denv_col >= 0`` guard was False.
    """
    import csv
    import json

    from causal_rl.plotting.headline_regression import make_headline_regression

    rng = np.random.default_rng(123)
    results_dir = tmp_path / "results"

    eval_cols = [
        "step", "cell", "env_name", "algorithm", "behaviour_policy", "seed",
        "eval_return_mean", "eval_oracle_return_mean", "eval_return_std",
        "eval_oracle_return_std", "id_status", "delta_tv", "bound_width_mean",
        "min_propensity", "ess_ratio", "overlap_at_1e-2", "tail_mass_top10pct",
        "propensity_calibration_ece", "bias_strength", "alpha_conf",
        "expose_z", "pi_b_known", "delta_tv_ci_lo", "delta_tv_ci_hi",
        "delta_kl", "delta_chi2", "delta_sup", "delta_mmd2", "delta_ks",
    ]
    perturbed_cols = [
        "seed", "cell", "env", "algorithm", "behaviour",
        "eval_perturbed_return_mean", "eval_perturbed_return_std",
        "D_env_KS", "eps_T", "eps_R",
    ]
    strata = {"id": 2.0, "partial_id": 1.0, "non_id": 0.0}
    n_per = 60
    run_idx = 0
    for status, slope in strata.items():
        for i in range(n_per):
            run_dir = results_dir / f"run_{run_idx}"
            run_dir.mkdir(parents=True)
            run_idx += 1

            delta_tv = float(rng.uniform(0.0, 0.5))
            oracle_return = float(rng.uniform(0.5, 1.0))
            G = slope * delta_tv + float(rng.normal(0, 0.2))  # noqa: N806

            eval_row = {
                "step": 100, "cell": 1, "env_name": "tabular-sepsis-v0",
                "algorithm": "dqn", "behaviour_policy": "uniform", "seed": i,
                "eval_return_mean": oracle_return - G,
                "eval_oracle_return_mean": oracle_return,
                "eval_return_std": 0.1, "eval_oracle_return_std": 0.1,
                "id_status": status, "delta_tv": delta_tv,
                "bound_width_mean": float(rng.uniform(0.1, 0.5)),
                "min_propensity": float(rng.uniform(0.05, 0.5)),
                "ess_ratio": float(rng.uniform(0.1, 1.0)),
                "overlap_at_1e-2": float(rng.uniform(0.5, 1.0)),
                "tail_mass_top10pct": float(rng.uniform(0.05, 0.3)),
                "propensity_calibration_ece": float(rng.uniform(0.0, 0.1)),
                "bias_strength": 1.0, "alpha_conf": 2.0,
                "expose_z": 1.0, "pi_b_known": 0.0,
                "delta_tv_ci_lo": delta_tv * 0.9,
                "delta_tv_ci_hi": delta_tv * 1.1,
                "delta_kl": delta_tv * 1.2, "delta_chi2": delta_tv * 0.8,
                "delta_sup": delta_tv * 0.7, "delta_mmd2": delta_tv * 0.6,
                "delta_ks": delta_tv * 0.5,
            }
            with (run_dir / "eval.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=eval_cols)
                writer.writeheader()
                writer.writerow(eval_row)
            # D_env_KS is constant 0 → dropped as constant feature → triggers fallback.
            with (run_dir / "eval_perturbed.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=perturbed_cols)
                writer.writeheader()
                writer.writerow({
                    "seed": i, "cell": 1, "env": "tabular-sepsis-v0",
                    "algorithm": "dqn", "behaviour": "uniform",
                    "eval_perturbed_return_mean": oracle_return - G - float(rng.normal(0, 0.05)),
                    "eval_perturbed_return_std": 0.1,
                    "D_env_KS": 0.0, "eps_T": 0.2, "eps_R": 0.1,
                })
            (run_dir / "meta.json").write_text(json.dumps({"offline_transitions": 50000}))

    out = tmp_path / "out"
    make_headline_regression(results_dir, out)
    pdf = out / "headline_pure_divergence.pdf"
    assert pdf.exists(), (
        "headline_pure_divergence.pdf was not written — fallback should have "
        "kicked in when D_env_KS is dropped"
    )
    assert pdf.stat().st_size > 1024, (
        f"headline_pure_divergence.pdf is suspiciously small "
        f"({pdf.stat().st_size} bytes) — figure may have rendered empty"
    )
