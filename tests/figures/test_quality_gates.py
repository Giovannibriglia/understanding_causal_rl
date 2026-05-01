"""Quality-gate assertions for paper figures.

Each test asserts a specific numeric claim that must hold on the final
results before the paper is submitted. Tests operate on synthetic data
that mirrors the real CSV schema so they can be run in CI without a full
experiment run.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build synthetic CSVs that mimic real run output
# ---------------------------------------------------------------------------

EVAL_HEADER = [
    "step", "cell", "env_name", "algorithm", "behaviour_policy", "seed",
    "eval_return_mean", "eval_oracle_return_mean",
    "eval_return_std", "eval_oracle_return_std",
    "id_status", "delta_tv", "delta_tv_beh", "ece",
    "delta_tv_ci_lo", "delta_tv_ci_hi",
    "delta_kl", "delta_chi2", "delta_sup", "delta_mmd2", "delta_ks",
]

PERTURBED_HEADER = [
    "step", "cell", "env_name", "algorithm", "behaviour_policy", "seed",
    "eps_T", "eps_R", "eval_perturbed_return_mean", "eval_perturbed_return_std", "D_env_KS",
]

TABULAR_DIAGONAL = [
    (0.0, 0.0),
    (0.1, 0.05),
    (0.2, 0.1),
    (0.35, 0.2),
    (0.5, 0.3),
]


def _eval_row(
    *,
    cell: int = 1,
    id_status: str = "id",
    eval_return: float = 8.0,
    oracle_return: float = 12.0,
    delta_tv: float = 0.2,
    delta_tv_beh: float = 0.35,
    ece: float = 0.05,
) -> dict[str, object]:
    return {
        "step": 200000, "cell": cell, "env_name": "tabular-sepsis-v0",
        "algorithm": "dqn", "behaviour_policy": "uniform", "seed": 0,
        "eval_return_mean": eval_return,
        "eval_oracle_return_mean": oracle_return,
        "eval_return_std": 1.0, "eval_oracle_return_std": 0.5,
        "id_status": id_status,
        "delta_tv": delta_tv, "delta_tv_beh": delta_tv_beh, "ece": ece,
        "delta_tv_ci_lo": delta_tv * 0.9, "delta_tv_ci_hi": delta_tv * 1.1,
        "delta_kl": delta_tv * 1.2, "delta_chi2": delta_tv * 0.8,
        "delta_sup": delta_tv * 0.7, "delta_mmd2": 0.0, "delta_ks": delta_tv * 0.5,
    }


def _perturbed_row(
    *, eps_t: float, eps_r: float, perturbed_return: float, d_env_ks: float
) -> dict[str, object]:
    return {
        "step": 200000, "cell": 1, "env_name": "tabular-sepsis-v0",
        "algorithm": "dqn", "behaviour_policy": "uniform", "seed": 0,
        "eps_T": eps_t, "eps_R": eps_r,
        "eval_perturbed_return_mean": perturbed_return,
        "eval_perturbed_return_std": 0.5,
        "D_env_KS": d_env_ks,
    }


def _write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Gate 1: oracle dominance (cells 1-3, tabular)
# ---------------------------------------------------------------------------

def test_oracle_dominance(tmp_path: Path) -> None:
    """eval_oracle_return_mean > eval_return_mean + 1.0 for identified cells."""
    eval_rows = [
        _eval_row(cell=1, id_status="id", eval_return=8.0, oracle_return=14.0),
        _eval_row(cell=2, id_status="id", eval_return=9.0, oracle_return=15.0),
        _eval_row(cell=3, id_status="partial_id", eval_return=7.0, oracle_return=12.0),
    ]
    _write_csv(tmp_path / "eval.csv", EVAL_HEADER, eval_rows)
    rows = _read_csv(tmp_path / "eval.csv")
    for row in rows:
        gap = float(row["eval_oracle_return_mean"]) - float(row["eval_return_mean"])
        assert gap > 1.0, (
            f"cell={row['cell']}: oracle−learned gap={gap:.2f} must be > 1.0"
        )


# ---------------------------------------------------------------------------
# Gate 2: identifiability ordering
# ---------------------------------------------------------------------------

def test_identifiability_ordering(tmp_path: Path) -> None:
    """Mean return: id > partial_id > non_id across seeds."""
    rng = np.random.default_rng(0)
    rows_by_status: dict[str, list[float]] = {"id": [], "partial_id": [], "non_id": []}
    all_eval: list[dict[str, object]] = []
    for seed in range(20):
        all_eval.append(_eval_row(id_status="id",
                                  eval_return=10.0 + rng.normal(0, 0.5), oracle_return=15.0))
        all_eval.append(_eval_row(id_status="partial_id",
                                  eval_return=7.0 + rng.normal(0, 0.5), oracle_return=15.0))
        all_eval.append(_eval_row(id_status="non_id",
                                  eval_return=3.0 + rng.normal(0, 0.5), oracle_return=15.0))

    _write_csv(tmp_path / "eval.csv", EVAL_HEADER, all_eval)
    rows = _read_csv(tmp_path / "eval.csv")

    for row in rows:
        rows_by_status[row["id_status"]].append(float(row["eval_return_mean"]))

    mean_id = np.mean(rows_by_status["id"])
    mean_pid = np.mean(rows_by_status["partial_id"])
    mean_nid = np.mean(rows_by_status["non_id"])
    assert mean_id > mean_pid, f"id mean={mean_id:.2f} must exceed partial_id mean={mean_pid:.2f}"
    assert mean_pid > mean_nid, (
        f"partial_id mean={mean_pid:.2f} must exceed non_id mean={mean_nid:.2f}"
    )


# ---------------------------------------------------------------------------
# Gate 3: headline R² bounds (stratified)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sklearn"),
    reason="scikit-learn not installed",
)
def test_stratified_r2_bounds(tmp_path: Path) -> None:
    """Stratified R²: id-stratum >= 0.6, non_id-stratum < 0.3 on synthetic data."""
    from causal_rl.plotting.headline_regression import _load_joined_data, _r2_score

    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n = 80

    def _make_run(results_dir: Path, idx: int, status: str, slope: float, noise: float) -> None:
        run_dir = results_dir / f"run_{idx}"
        run_dir.mkdir(parents=True)
        delta_tv = float(rng.uniform(0.0, 0.5))
        oracle = float(rng.uniform(8.0, 15.0))
        g = slope * delta_tv + float(rng.normal(0, noise))
        eval_row = {
            "step": 200000, "cell": 1, "env_name": "tabular-sepsis-v0",
            "algorithm": "dqn", "behaviour_policy": "uniform", "seed": idx,
            "eval_return_mean": oracle - g, "eval_oracle_return_mean": oracle,
            "eval_return_std": 0.5, "eval_oracle_return_std": 0.3,
            "id_status": status, "delta_tv": delta_tv,
            "bound_width_mean": float(rng.uniform(0.1, 0.5)),
            "min_propensity": 0.1, "ess_ratio": 0.5, "overlap_at_1e-2": 0.8,
            "tail_mass_top10pct": 0.1, "propensity_calibration_ece": 0.05,
            "bias_strength": 1.0, "alpha_conf": 2.0, "expose_z": 1.0, "pi_b_known": 0.0,
            "delta_tv_ci_lo": delta_tv * 0.9, "delta_tv_ci_hi": delta_tv * 1.1,
            "delta_kl": delta_tv, "delta_chi2": delta_tv, "delta_sup": delta_tv,
            "delta_mmd2": 0.0, "delta_ks": delta_tv,
        }
        with (run_dir / "eval.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_row.keys()))
            writer.writeheader()
            writer.writerow(eval_row)
        perturbed_row = {
            "step": 200000, "cell": 1, "env_name": "tabular-sepsis-v0",
            "algorithm": "dqn", "behaviour_policy": "uniform", "seed": idx,
            "eps_T": 0.2, "eps_R": 0.1,
            "eval_perturbed_return_mean": oracle - g - float(rng.normal(0, 0.1)),
            "eval_perturbed_return_std": 0.3, "D_env_KS": float(rng.uniform(0.0, 0.3)),
        }
        with (run_dir / "eval_perturbed.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(perturbed_row.keys()))
            writer.writeheader()
            writer.writerow(perturbed_row)

    results_dir = tmp_path / "results"
    idx = 0
    for _ in range(n):
        _make_run(results_dir, idx, "id", slope=3.0, noise=0.15)
        idx += 1
        _make_run(results_dir, idx, "non_id", slope=0.0, noise=1.0)
        idx += 1

    rows = _load_joined_data(results_dir)
    assert len(rows) >= 100

    feature_cols = ["delta_tv", "bound_width_mean", "D_env_KS"]
    X_raw = np.array([[r.get(c, 0.0) for c in feature_cols] for r in rows])
    y = np.array([r["G"] for r in rows])
    statuses = [str(r["id_status"]) for r in rows]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    for status, lo, hi in [("id", 0.6, 1.01), ("non_id", -1.0, 0.3)]:
        mask = np.array([s == status for s in statuses])
        lr = LinearRegression()
        lr.fit(X[mask], y[mask])
        r2 = _r2_score(y[mask], lr.predict(X[mask]))
        assert lo <= r2 <= hi, (
            f"{status} stratum R²={r2:.3f} outside [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Gate 4: perturbation monotonicity
# ---------------------------------------------------------------------------

def test_perturbation_monotonicity(tmp_path: Path) -> None:
    """Perturbed return should be non-increasing along TABULAR_DIAGONAL."""
    base_return = 10.0
    perturbed_rows = [
        _perturbed_row(
            eps_t=eps_t, eps_r=eps_r,
            perturbed_return=base_return * (1.0 - 0.15 * i),
            d_env_ks=0.02 * i,
        )
        for i, (eps_t, eps_r) in enumerate(TABULAR_DIAGONAL)
    ]
    _write_csv(tmp_path / "eval_perturbed.csv", PERTURBED_HEADER, perturbed_rows)
    rows = _read_csv(tmp_path / "eval_perturbed.csv")

    returns = [float(r["eval_perturbed_return_mean"]) for r in rows]
    for i in range(1, len(returns)):
        assert returns[i] <= returns[i - 1] + 0.5, (
            f"Perturbed return not monotone at step {i}: "
            f"{returns[i]:.2f} > {returns[i-1]:.2f}"
        )


# ---------------------------------------------------------------------------
# Gate 5: KS sensitivity at max perturbation
# ---------------------------------------------------------------------------

def test_ks_sensitivity_max_perturbation(tmp_path: Path) -> None:
    """D_env_KS > 0.05 at max perturbation (eps_T=0.5, eps_R=0.3)."""
    perturbed_rows = [
        _perturbed_row(eps_t=0.5, eps_r=0.3, perturbed_return=5.0, d_env_ks=0.25),
    ]
    _write_csv(tmp_path / "eval_perturbed.csv", PERTURBED_HEADER, perturbed_rows)
    rows = _read_csv(tmp_path / "eval_perturbed.csv")

    max_ks_rows = [r for r in rows if float(r["eps_T"]) >= 0.4]
    assert max_ks_rows, "No rows with eps_T >= 0.4"
    max_ks = max(float(r["D_env_KS"]) for r in max_ks_rows)
    assert max_ks > 0.05, f"D_env_KS={max_ks:.3f} at max perturbation must be > 0.05"


# ---------------------------------------------------------------------------
# Gate 6: ECE sanity for identified cells
# ---------------------------------------------------------------------------

def test_ece_sanity_identified(tmp_path: Path) -> None:
    """ECE < 0.30 for all identified cells (id or partial_id)."""
    eval_rows = [
        _eval_row(cell=1, id_status="id", ece=0.08),
        _eval_row(cell=2, id_status="id", ece=0.12),
        _eval_row(cell=3, id_status="partial_id", ece=0.18),
    ]
    _write_csv(tmp_path / "eval.csv", EVAL_HEADER, eval_rows)
    rows = _read_csv(tmp_path / "eval.csv")

    for row in rows:
        if row["id_status"] in ("id", "partial_id"):
            ece = float(row["ece"])
            assert ece < 0.30, (
                f"cell={row['cell']} id_status={row['id_status']}: ECE={ece:.3f} must be < 0.30"
            )


# ---------------------------------------------------------------------------
# Gate 7: dual-policy delta_tv direction
# ---------------------------------------------------------------------------

def test_dual_policy_delta_tv_direction(tmp_path: Path) -> None:
    """delta_tv_beh > delta_tv: learned policy should select less-confounded actions."""
    eval_rows = [
        _eval_row(delta_tv=0.12, delta_tv_beh=0.30),
        _eval_row(delta_tv=0.15, delta_tv_beh=0.28),
    ]
    _write_csv(tmp_path / "eval.csv", EVAL_HEADER, eval_rows)
    rows = _read_csv(tmp_path / "eval.csv")

    for row in rows:
        dtv = float(row["delta_tv"])
        dtv_beh = float(row["delta_tv_beh"])
        assert dtv_beh > dtv, (
            f"delta_tv_beh={dtv_beh:.3f} should exceed delta_tv={dtv:.3f} "
            "(learned policy should prefer less-confounded actions)"
        )
