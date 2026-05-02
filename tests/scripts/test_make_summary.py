"""Tests for scripts/make_summary.py.

Builds a minimal synthetic results directory and checks that make_summary
produces a valid summary.json with the required schema keys.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from make_summary import make_summary  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: synthetic results directory
# ---------------------------------------------------------------------------

_EVAL_COLS = [
    "step", "episode", "wall_time", "cell", "env_name", "env_fidelity",
    "algorithm", "behaviour_policy", "seed",
    "eval_return_mean", "eval_return_std",
    "delta_tv", "delta_tv_ci_lo", "delta_tv_ci_hi",
    "delta_kl", "delta_chi2", "delta_sup", "delta_mmd2", "delta_ks",
    "min_propensity", "ess_ratio", "overlap_at_1e-2", "tail_mass_top10pct",
    "propensity_calibration_ece",
    "bound_width_mean", "bound_width_max", "n_arms_with_informative_bound",
    "alpha_conf", "bias_strength", "expose_z", "pi_b_known", "beh_depends_on_u",
    "id_status",
    "eval_oracle_return_mean", "eval_oracle_return_std",
    "delta_tv_beh", "ece",
]

_TRAIN_COLS = ["step", "episode", "wall_time", "cell", "env_name", "env_fidelity",
               "algorithm", "behaviour_policy", "seed", "train_return_mean",
               "train_return_std", "policy_loss", "value_loss", "td_loss",
               "entropy", "kl", "delta_tv", "delta_tv_ci_lo", "delta_tv_ci_hi",
               "delta_kl", "delta_chi2", "delta_sup", "delta_mmd2", "delta_ks",
               "delta_tv_marginal", "delta_tv_conditional", "min_propensity",
               "ess_ratio", "overlap_at_1e-2", "tail_mass_top10pct",
               "propensity_calibration_ece", "bound_width_mean", "bound_width_max",
               "n_arms_with_informative_bound", "delta_tv_train_policy",
               "delta_tv_train_policy_ci_lo", "delta_tv_train_policy_ci_hi",
               "alpha_conf", "bias_strength", "expose_z", "pi_b_known",
               "beh_depends_on_u", "id_status", "lr", "grad_norm"]

_PERTURBED_COLS = [
    "step", "episode", "wall_time", "cell", "env_name", "algorithm",
    "behaviour_policy", "seed", "eps_T", "eps_R",
    "eval_perturbed_return_mean", "eval_perturbed_return_std", "D_env_KS",
]


def _make_eval_row(cell: int, algo: str, beh: str, seed: int, id_status: str) -> dict:
    return {
        "step": "500", "episode": "10", "wall_time": str(1000.0 + seed * 10),
        "cell": str(cell), "env_name": "tabular-sepsis-v0", "env_fidelity": "high",
        "algorithm": algo, "behaviour_policy": beh, "seed": str(seed),
        "eval_return_mean": "12.5", "eval_return_std": "1.2",
        "delta_tv": "0.03", "delta_tv_ci_lo": "0.02", "delta_tv_ci_hi": "0.04",
        "delta_kl": "0.01", "delta_chi2": "0.01", "delta_sup": "0.02",
        "delta_mmd2": "0.0", "delta_ks": "0.0",
        "min_propensity": "0.4", "ess_ratio": "0.9",
        "overlap_at_1e-2": "0.85", "tail_mass_top10pct": "0.1",
        "propensity_calibration_ece": "nan",
        "bound_width_mean": "nan", "bound_width_max": "nan",
        "n_arms_with_informative_bound": "nan",
        "alpha_conf": "0.0", "bias_strength": "1.0",
        "expose_z": "1" if cell <= 2 else "0",
        "pi_b_known": "1" if cell in (1, 3) else "0",
        "beh_depends_on_u": "0",
        "id_status": id_status,
        "eval_oracle_return_mean": "18.0", "eval_oracle_return_std": "0.5",
        "delta_tv_beh": "0.03", "ece": "0.05",
    }


def _write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _build_fake_results(tmp_path: Path) -> Path:
    results_dir = tmp_path / "smoke_fake_20260101_000000"
    results_dir.mkdir()

    # Two cells × two algos × two seeds = 8 runs.
    runs = [
        (1, "dqn", "uniform", 0, "id"),
        (1, "dqn", "uniform", 1, "id"),
        (2, "cql", "uniform", 0, "id"),
        (2, "cql", "uniform", 1, "partial_id"),
        (3, "dqn", "uniform", 0, "partial_id"),
        (3, "cql", "uniform", 1, "partial_id"),
        (4, "dqn", "uniform", 0, "partial_id"),
        (4, "cql", "uniform", 1, "non_id"),
    ]

    for cell, algo, beh, seed, id_status in runs:
        run_name = f"cell{cell}_tabular-sepsis-v0_{algo}_{beh}_seed{seed}"
        run_dir = results_dir / run_name
        run_dir.mkdir()

        eval_row = _make_eval_row(cell, algo, beh, seed, id_status)
        _write_csv(run_dir / "eval.csv", _EVAL_COLS, [eval_row, eval_row])

        train_row = {c: "0.0" for c in _TRAIN_COLS}
        train_row.update({"step": "500", "episode": "5", "wall_time": str(900.0 + seed * 5)})
        _write_csv(run_dir / "train.csv", _TRAIN_COLS, [train_row])

        perturbed_rows = [
            {"step": "500", "episode": "10", "wall_time": "1001.0",
             "cell": str(cell), "env_name": "tabular-sepsis-v0",
             "algorithm": algo, "behaviour_policy": beh, "seed": str(seed),
             "eps_T": "0.1", "eps_R": "0.1",
             "eval_perturbed_return_mean": "10.0", "eval_perturbed_return_std": "1.0",
             "D_env_KS": "0.05"},
        ]
        _write_csv(run_dir / "eval_perturbed.csv", _PERTURBED_COLS, perturbed_rows)

        meta = {
            "seed": seed, "cell": cell, "env": "tabular-sepsis-v0",
            "algorithm": algo, "behaviour": beh, "device": "cpu",
            "horizon": 20, "rollout_horizon": 20, "n_envs": 16,
            "total_frames": 500, "total_transitions": 8000,
            "n_checkpoints_train": 5, "n_checkpoints_eval": 2,
            "id_status": id_status, "cli": "run_single.py ...",
            "git_commit": "abc1234", "python": "3.11", "platform": "Linux",
            "torch": "2.0",
        }
        (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Write manifest.csv
    manifest_fields = ["seed", "alpha_conf", "cell", "env", "algorithm", "behaviour", "output_path"]
    with (results_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for cell, algo, beh, seed, _ in runs:
            writer.writerow({
                "seed": seed, "alpha_conf": "0.0", "cell": cell,
                "env": "tabular-sepsis-v0", "algorithm": algo,
                "behaviour": beh,
                "output_path": str(results_dir / f"cell{cell}_tabular-sepsis-v0_{algo}_{beh}_seed{seed}"),
            })

    return results_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_results(tmp_path: Path) -> Path:
    return _build_fake_results(tmp_path)


def test_make_summary_creates_file(fake_results: Path) -> None:
    out = make_summary(fake_results)
    assert out.exists(), "summary.json was not created"


def test_make_summary_top_level_keys(fake_results: Path) -> None:
    make_summary(fake_results)
    data = json.loads((fake_results / "summary.json").read_text())
    required = {
        "run_id", "timestamp", "config", "n_runs", "n_runs_failed",
        "wall_time_seconds", "git_sha",
        "per_cell_summary", "per_algorithm_summary",
        "per_cell_per_algorithm", "per_cell_per_env",
        "headline_regression", "claims_evidence",
    }
    assert required <= data.keys(), f"Missing keys: {required - data.keys()}"


def test_make_summary_n_runs(fake_results: Path) -> None:
    make_summary(fake_results)
    data = json.loads((fake_results / "summary.json").read_text())
    assert data["n_runs"] == 8


def test_make_summary_per_cell_keys(fake_results: Path) -> None:
    make_summary(fake_results)
    data = json.loads((fake_results / "summary.json").read_text())
    per_cell = data["per_cell_summary"]
    assert set(per_cell.keys()) == {"1", "2", "3", "4"}
    for cell_data in per_cell.values():
        assert "shorthand" in cell_data
        assert "eval_return_mean" in cell_data
        assert "gap_to_oracle_mean" in cell_data
        assert "delta_tv_mean" in cell_data
        assert "id_statuses" in cell_data


def test_make_summary_claims_present(fake_results: Path) -> None:
    make_summary(fake_results)
    data = json.loads((fake_results / "summary.json").read_text())
    claims = data["claims_evidence"]
    expected_claims = {
        "claim_1_id_separation",
        "claim_2_dtv_separates_strata",
        "claim_3_behaviour_silent_at_alpha_zero",
        "claim_4_pooled_r2_in_range",
        "claim_5_non_id_populated",
    }
    assert expected_claims <= claims.keys()
    for claim in claims.values():
        assert "verdict" in claim


def test_make_summary_no_nan_in_json(fake_results: Path) -> None:
    """JSON must not contain bare NaN (invalid JSON)."""
    make_summary(fake_results)
    raw = (fake_results / "summary.json").read_text()
    # Valid JSON parseable without error
    data = json.loads(raw)
    assert isinstance(data, dict)
    # Must not contain literal NaN token (json.loads would have raised already,
    # but double-check the raw text doesn't contain it as a bare token).
    assert "NaN" not in raw


def test_make_summary_custom_output(fake_results: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "custom_summary.json"
    result = make_summary(fake_results, out_path)
    assert result == out_path
    assert out_path.exists()
