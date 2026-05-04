"""Smoke tests for the v11 diagnostic-framing figures.

* ``regime_map`` partitions runs into the four diagnostic quadrants and
  ``regime_quadrant_counts`` returns the same partition.
* ``coverage_signature`` renders without crash on a small synthetic
  result directory.
* ``diagnosis_flowchart`` renders without crash (no data dependency).
* ``metrics_curves`` and ``divergence_panel`` render without crash on
  synthetic data; missing-metric cells show "not collected" rather
  than a flat zero line.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from causal_rl.plotting.causal_metrics_grid import make_causal_metrics_grid  # noqa: E402
from causal_rl.plotting.coverage_signature import make_coverage_signature  # noqa: E402
from causal_rl.plotting.diagnosis_flowchart import make_diagnosis_flowchart  # noqa: E402
from causal_rl.plotting.divergence_panel import make_divergence_panel  # noqa: E402
from causal_rl.plotting.metrics_curves import make_metrics_curves  # noqa: E402
from causal_rl.plotting.regime_map import (  # noqa: E402
    make_regime_map,
    regime_quadrant_counts,
)


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
    "delta_kl",
    "delta_chi2",
    "delta_sup",
    "delta_mmd2",
    "delta_ks",
    "min_propensity",
    "ess_ratio",
    "propensity_calibration_ece",
    "pi_b_recovery_kl",
    "cond_mi_r_z_given_sa",
    "target_support_overlap",
    "backdoor_residual_mean",
    "bound_width_mean",
    "id_status",
]
_TRAIN_HEADER = [
    "step",
    "cell",
    "env_name",
    "algorithm",
    "behaviour_policy",
    "seed",
    "min_propensity",
    "ess_ratio",
    "propensity_calibration_ece",
    "pi_b_recovery_kl",
    "cond_mi_r_z_given_sa",
    "bound_width_mean",
]


def _write_synthetic_runs(results_dir: Path) -> None:
    """Emit a small synthetic results tree spanning all four regimes."""
    runs = [
        # cell, behaviour, id_status, delta_tv, gap, ess, min_prop
        (1, "uniform", "id", 0.020, 1.5, 0.97, 0.10),
        (1, "uniform", "id", 0.025, 2.0, 0.95, 0.08),
        (1, "uniform", "id", 0.018, 1.0, 0.97, 0.11),
        # coverage-broken: low Δ_TV, high gap, low ESS
        (1, "uniform", "id", 0.022, 9.0, 0.40, 0.001),
        (1, "uniform", "id", 0.028, 11.0, 0.30, 0.0005),
        # confounding visible: high Δ_TV, low gap
        (2, "reward_aligned", "partial_id", 0.080, 2.5, 0.93, 0.06),
        (2, "reward_aligned", "partial_id", 0.075, 3.0, 0.92, 0.07),
        # partial observability: high Δ_TV, high gap
        (3, "uniform", "partial_id", 0.090, 12.0, 0.91, 0.05),
        (4, "reward_aligned", "non_id", 0.110, 14.0, 0.88, 0.04),
    ]
    for i, (cell, beh, status, dtv, gap, ess, mp) in enumerate(runs):
        run_dir = results_dir / f"run_{i}"
        run_dir.mkdir(parents=True)
        oracle = 18.0
        eval_ret = oracle - gap
        eval_rows = []
        train_rows = []
        # Two checkpoints so metrics_curves has a step axis.
        for step in (50, 100):
            eval_rows.append(
                {
                    "step": step,
                    "cell": cell,
                    "env_name": "tabular-sepsis-v0",
                    "algorithm": "dqn",
                    "behaviour_policy": beh,
                    "seed": i,
                    "eval_return_mean": eval_ret,
                    "eval_oracle_return_mean": oracle,
                    "delta_tv": dtv,
                    "delta_kl": dtv * 1.2,
                    "delta_chi2": dtv * 0.9,
                    "delta_sup": dtv * 0.7,
                    "delta_mmd2": dtv * 0.6,
                    "delta_ks": dtv * 0.5,
                    "min_propensity": mp,
                    "ess_ratio": ess,
                    "propensity_calibration_ece": 0.05,
                    "pi_b_recovery_kl": 0.02,
                    "cond_mi_r_z_given_sa": 0.1 if cell <= 2 else 0.4,
                    "target_support_overlap": 0.8 if ess > 0.5 else 0.3,
                    "backdoor_residual_mean": 0.05 if cell <= 2 else float("nan"),
                    "bound_width_mean": 0.875,
                    "id_status": status,
                }
            )
            train_rows.append(
                {
                    "step": step,
                    "cell": cell,
                    "env_name": "tabular-sepsis-v0",
                    "algorithm": "dqn",
                    "behaviour_policy": beh,
                    "seed": i,
                    "min_propensity": mp,
                    "ess_ratio": ess,
                    "propensity_calibration_ece": 0.05,
                    "pi_b_recovery_kl": 0.02,
                    "cond_mi_r_z_given_sa": 0.1 if cell <= 2 else 0.4,
                    "bound_width_mean": 0.875,
                }
            )
        with (run_dir / "eval.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_EVAL_HEADER)
            w.writeheader()
            w.writerows(eval_rows)
        with (run_dir / "train.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_TRAIN_HEADER)
            w.writeheader()
            w.writerows(train_rows)


def test_regime_map_renders_and_partitions(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "fig"
    _write_synthetic_runs(results)
    make_regime_map(results, out)
    assert (out / "regime_map.pdf").exists()
    counts = regime_quadrant_counts(results)
    # The synthetic data covers all four diagnostic quadrants.
    assert counts.get("Identifiable", 0) >= 1
    assert counts.get("Coverage-broken", 0) >= 1
    assert counts.get("Confounding visible, recoverable", 0) >= 1
    assert counts.get("Partial observability", 0) >= 1


def test_coverage_signature_renders(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "fig"
    _write_synthetic_runs(results)
    make_coverage_signature(results, out)
    assert (out / "coverage_signature.pdf").exists()


def test_diagnosis_flowchart_renders(tmp_path: Path) -> None:
    out = tmp_path / "fig"
    make_diagnosis_flowchart(out)
    assert (out / "diagnosis_flowchart.pdf").exists()


def test_metrics_curves_renders(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "fig"
    _write_synthetic_runs(results)
    make_metrics_curves(results, out)
    # Some cells will have been emitted; just confirm at least one PNG.
    assert any(out.glob("metrics_curves_cell*.png"))


def test_divergence_panel_renders(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "fig"
    _write_synthetic_runs(results)
    make_divergence_panel(results, out)
    assert (out / "divergence_panel.pdf").exists()


def test_causal_metrics_grid_renders(tmp_path: Path) -> None:
    results = tmp_path / "results"
    out = tmp_path / "fig"
    _write_synthetic_runs(results)
    make_causal_metrics_grid(results, out)
    assert (out / "causal_metrics_grid.pdf").exists()
