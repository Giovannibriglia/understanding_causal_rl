"""Aggregate a results directory into a single summary.json.

Usage::

    python scripts/make_summary.py --results results/<run_id>
    python scripts/make_summary.py --results results/<run_id> --output my_summary.json

Writes ``<results_dir>/summary.json`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys  # noqa: E402

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_CELL_SHORTHANDS = {1: "mdp_known", 2: "mdp_unknown", 3: "pomdp_known", 4: "pomdp_unknown"}
_ALL_ENVS = ["tabular-sepsis-v0", "continuous-ward-v0"]
_FEATURE_COLS = [
    "delta_tv",
    "bound_width_mean",
    "min_propensity",
    "ess_ratio",
    "overlap_at_1e-2",
    "tail_mass_top10pct",
    "propensity_calibration_ece",
    "pi_b_recovery_kl",
    "backdoor_residual_mean",
    "target_support_overlap",
    "cond_mi_r_z_given_sa",
    "bias_strength",
    "id_status_ordinal",
    "D_env_KS",
    "D_bisim",
    "alpha_conf",
    "expose_z",
    "pi_b_known",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(ROOT),
        ).strip()
    except Exception:
        return "unknown"


def _read_last_row(path: Path) -> dict[str, str] | None:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def _read_all_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sf(val: object) -> float | None:
    """Safe float: returns None for empty/NaN/missing."""
    if val is None or val == "" or val == "nan":
        return None
    try:
        v = float(val)  # type: ignore[arg-type]
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _mean(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _std(vals: list[float | None]) -> float | None:
    clean = [v for v in vals if v is not None]
    return float(np.std(clean)) if len(clean) > 1 else None


def _nan_to_null(obj: Any) -> Any:
    """Recursively convert float NaN/Inf to None for JSON serialisation."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _collect_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Walk results_dir and collect the last eval row + meta for each run."""
    runs: list[dict[str, Any]] = []
    for eval_path in sorted(results_dir.rglob("eval.csv")):
        run_dir = eval_path.parent
        last_eval = _read_last_row(eval_path)
        if last_eval is None:
            continue

        meta: dict[str, Any] = {}
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        cell = int(meta.get("cell") or last_eval.get("cell") or 0)
        env = str(meta.get("env") or last_eval.get("env_name") or "unknown")
        algo = str(meta.get("algorithm") or last_eval.get("algorithm") or "unknown")
        beh = str(meta.get("behaviour") or last_eval.get("behaviour_policy") or "unknown")
        seed = int(meta.get("seed") or last_eval.get("seed") or 0)
        horizon: int | None = None
        if meta.get("horizon") is not None:
            try:
                horizon = int(meta["horizon"])
            except (TypeError, ValueError):
                horizon = None

        eval_ret = _sf(last_eval.get("eval_return_mean"))
        oracle_ret = _sf(last_eval.get("eval_oracle_return_mean"))
        gap = (
            (oracle_ret - eval_ret)
            if (eval_ret is not None and oracle_ret is not None)
            else None
        )

        runs.append(
            {
                "cell": cell,
                "env": env,
                "algorithm": algo,
                "behaviour": beh,
                "seed": seed,
                "horizon": horizon,
                "id_status": str(last_eval.get("id_status") or "non_id"),
                "alpha_conf": _sf(last_eval.get("alpha_conf")),
                "bias_strength": _sf(last_eval.get("bias_strength")),
                "eval_return_mean": eval_ret,
                "eval_oracle_return_mean": oracle_ret,
                "gap_to_oracle": gap,
                "delta_tv": _sf(last_eval.get("delta_tv")),
                "bound_width_mean": _sf(last_eval.get("bound_width_mean")),
                "min_propensity": _sf(last_eval.get("min_propensity")),
                "ess_ratio": _sf(last_eval.get("ess_ratio")),
                "propensity_calibration_ece": _sf(
                    last_eval.get("propensity_calibration_ece")
                ),
                "pi_b_recovery_kl": _sf(last_eval.get("pi_b_recovery_kl")),
                "backdoor_residual_mean": _sf(last_eval.get("backdoor_residual_mean")),
                "target_support_overlap": _sf(last_eval.get("target_support_overlap")),
                "cond_mi_r_z_given_sa": _sf(last_eval.get("cond_mi_r_z_given_sa")),
            }
        )
    return runs


def _group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
    id_counts = {
        k: sum(1 for r in group if r["id_status"] == k)
        for k in ["id", "partial_id", "non_id"]
    }
    return {
        "n_runs": len(group),
        "id_statuses": id_counts,
        "eval_return_mean": _mean([r["eval_return_mean"] for r in group]),
        "eval_return_std_across_runs": _std([r["eval_return_mean"] for r in group]),
        "eval_oracle_return_mean": _mean([r["eval_oracle_return_mean"] for r in group]),
        "gap_to_oracle_mean": _mean([r["gap_to_oracle"] for r in group]),
        "delta_tv_mean": _mean([r["delta_tv"] for r in group]),
        "delta_tv_std": _std([r["delta_tv"] for r in group]),
        "bound_width_mean": _mean([r["bound_width_mean"] for r in group]),
        "min_propensity_mean": _mean([r["min_propensity"] for r in group]),
        "ess_ratio_mean": _mean([r["ess_ratio"] for r in group]),
        "propensity_calibration_ece_mean": _mean(
            [r.get("propensity_calibration_ece") for r in group]
        ),
        "pi_b_recovery_kl_mean": _mean([r.get("pi_b_recovery_kl") for r in group]),
        "backdoor_residual_mean": _mean(
            [r.get("backdoor_residual_mean") for r in group]
        ),
        "target_support_overlap_mean": _mean(
            [r.get("target_support_overlap") for r in group]
        ),
        "cond_mi_r_z_given_sa_mean": _mean(
            [r.get("cond_mi_r_z_given_sa") for r in group]
        ),
    }


# ---------------------------------------------------------------------------
# Headline regression
# ---------------------------------------------------------------------------


def _compute_regression(results_dir: Path) -> dict[str, Any]:
    """Compute headline regression metrics (requires scikit-learn)."""
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
        from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
        from sklearn.model_selection import cross_val_score  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

        from causal_rl.identification.id_oracle import ID_STATUS_ORDER
    except ImportError as exc:
        return {"error": str(exc)}

    # Load joined data (eval + eval_perturbed per run).
    rows: list[dict[str, Any]] = []
    for eval_path in sorted(results_dir.rglob("eval.csv")):
        run_dir = eval_path.parent
        last_eval = _read_last_row(eval_path)
        if last_eval is None:
            continue
        perturbed_path = run_dir / "eval_perturbed.csv"
        if not perturbed_path.exists():
            continue
        perturbed_rows = _read_all_rows(perturbed_path)
        if not perturbed_rows:
            continue

        oracle_ret = _sf(last_eval.get("eval_oracle_return_mean")) or 0.0
        id_status = str(last_eval.get("id_status") or "non_id")
        for pr in perturbed_rows:
            pert_ret = _sf(pr.get("eval_perturbed_return_mean")) or 0.0
            rows.append(
                {
                    "G": oracle_ret - pert_ret,
                    "id_status": id_status,
                    "id_status_ordinal": ID_STATUS_ORDER.get(id_status, 2),
                    "delta_tv": _sf(last_eval.get("delta_tv")) or 0.0,
                    "bound_width_mean": _sf(last_eval.get("bound_width_mean")) or 0.0,
                    "min_propensity": _sf(last_eval.get("min_propensity")) or 0.0,
                    "ess_ratio": _sf(last_eval.get("ess_ratio")) or 0.0,
                    "overlap_at_1e-2": _sf(last_eval.get("overlap_at_1e-2")) or 0.0,
                    "tail_mass_top10pct": _sf(last_eval.get("tail_mass_top10pct")) or 0.0,
                    "propensity_calibration_ece": _sf(
                        last_eval.get("propensity_calibration_ece")
                    ),
                    "bias_strength": _sf(last_eval.get("bias_strength")) or 1.0,
                    "D_env_KS": _sf(pr.get("D_env_KS")) or 0.0,
                    "alpha_conf": _sf(last_eval.get("alpha_conf")) or 0.0,
                    "expose_z": _sf(last_eval.get("expose_z")) or 0.0,
                    "pi_b_known": _sf(last_eval.get("pi_b_known")) or 0.0,
                }
            )

    if len(rows) < 10:
        return {"error": f"insufficient data ({len(rows)} joined rows; need ≥10)"}

    feature_cols = [c for c in _FEATURE_COLS if c in rows[0]]
    X_raw = np.array(  # noqa: N806
        [[r.get(c) or 0.0 for c in feature_cols] for r in rows], dtype=np.float64
    )
    y = np.array([r["G"] for r in rows], dtype=np.float64)
    id_statuses = [r["id_status"] for r in rows]

    # Impute NaN propensity ECE with column median.
    if "propensity_calibration_ece" in feature_cols:
        ece_col = feature_cols.index("propensity_calibration_ece")
        mask = np.isnan(X_raw[:, ece_col])
        if mask.any() and (~mask).any():
            X_raw[mask, ece_col] = float(np.nanmedian(X_raw[:, ece_col]))
        elif mask.all():
            X_raw[:, ece_col] = 0.0

    # Drop constant features (fix 3.1).
    stds = X_raw.std(axis=0)
    keep = stds >= 1e-8
    dropped_features = [feature_cols[i] for i in range(len(feature_cols)) if not keep[i]]
    active_cols = [feature_cols[i] for i in range(len(feature_cols)) if keep[i]]
    X_active = X_raw[:, keep]  # noqa: N806

    scaler = StandardScaler()
    X = scaler.fit_transform(X_active)  # noqa: N806

    # Pooled linear regression.
    lin = LinearRegression().fit(X, y)
    y_pred = lin.predict(X)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_train = 1.0 - ss_res / max(ss_tot, 1e-12)
    n = len(y)
    r2_cv: float | None = None
    if n >= 6:
        cv = min(5, n // 2)
        scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
        r2_cv = float(np.mean(scores))

    # Random forest for feature importances.
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0).fit(X, y)

    # Per-stratum R².
    id_counts: dict[str, int] = {}
    strat: dict[str, dict[str, float | None]] = {}
    for status in ["id", "partial_id", "non_id"]:
        s_mask = np.array([s == status for s in id_statuses])
        id_counts[status] = int(s_mask.sum())
        if id_counts[status] < 10:
            strat[status] = {"r2_train": None, "r2_cv": None}
            continue
        lr_s = LinearRegression().fit(X[s_mask], y[s_mask])
        y_s, y_s_pred = y[s_mask], lr_s.predict(X[s_mask])
        ss_r = float(np.sum((y_s - y_s_pred) ** 2))
        ss_t = float(np.sum((y_s - y_s.mean()) ** 2))
        r2_s_train: float | None = 1.0 - ss_r / max(ss_t, 1e-12)
        r2_s_cv: float | None = None
        n_s = id_counts[status]
        if n_s >= 6:
            cv_s = min(5, n_s // 2)
            r2_s_cv = float(
                np.mean(
                    cross_val_score(LinearRegression(), X[s_mask], y[s_mask], cv=cv_s, scoring="r2")
                )
            )
        strat[status] = {"r2_train": r2_s_train, "r2_cv": r2_s_cv}

    top_features = sorted(
        [
            {
                "feature": f,
                "coef": float(lin.coef_[i]),
                "importance": float(rf.feature_importances_[i]),
            }
            for i, f in enumerate(active_cols)
        ],
        key=lambda x: -x["importance"],
    )[:5]

    return {
        "r2_pooled_train": r2_train,
        "r2_pooled_cv": r2_cv,
        "r2_id_train": strat["id"]["r2_train"],
        "r2_id_cv": strat["id"]["r2_cv"],
        "r2_partial_id_train": strat["partial_id"]["r2_train"],
        "r2_partial_id_cv": strat["partial_id"]["r2_cv"],
        "r2_non_id_train": strat["non_id"]["r2_train"],
        "r2_non_id_cv": strat["non_id"]["r2_cv"],
        "n_id": id_counts["id"],
        "n_partial_id": id_counts["partial_id"],
        "n_non_id": id_counts["non_id"],
        "dropped_features": dropped_features,
        "top_features_by_rf_importance": top_features,
    }


# ---------------------------------------------------------------------------
# Claims evidence
# ---------------------------------------------------------------------------


def _compute_claims(
    runs: list[dict[str, Any]], regression: dict[str, Any]
) -> dict[str, Any]:
    id_gaps = [r["gap_to_oracle"] for r in runs if r["id_status"] == "id" and r["gap_to_oracle"] is not None]
    pid_gaps = [r["gap_to_oracle"] for r in runs if r["id_status"] == "partial_id" and r["gap_to_oracle"] is not None]
    id_mean_gap = _mean(id_gaps)
    pid_mean_gap = _mean(pid_gaps)
    delta_gap = (
        (pid_mean_gap - id_mean_gap)
        if (id_mean_gap is not None and pid_mean_gap is not None)
        else None
    )
    c1_verdict = (
        "supported" if (delta_gap is not None and delta_gap > 0)
        else "refuted" if delta_gap is not None
        else "not_yet_testable"
    )

    id_dtv = _mean([r["delta_tv"] for r in runs if r["id_status"] == "id" and r["delta_tv"] is not None])
    pid_dtv = _mean([r["delta_tv"] for r in runs if r["id_status"] == "partial_id" and r["delta_tv"] is not None])
    ratio = (pid_dtv / max(id_dtv, 1e-8)) if (id_dtv is not None and pid_dtv is not None) else None
    c2_verdict = (
        "supported" if (ratio is not None and ratio > 1.2)
        else "refuted" if ratio is not None
        else "not_yet_testable"
    )

    alpha0_runs = [r for r in runs if r["alpha_conf"] is not None and r["alpha_conf"] <= 1e-6]
    max_spread: float | None = None
    typical_seed_std: float | None = None
    if alpha0_runs:
        beh_groups: dict = defaultdict(list)
        for r in alpha0_runs:
            beh_groups[(r["cell"], r["algorithm"], r["seed"])].append(r)
        spreads = []
        for g in beh_groups.values():
            if len(g) > 1:
                vals = [rr["eval_return_mean"] for rr in g if rr["eval_return_mean"] is not None]
                if vals:
                    spreads.append(max(vals) - min(vals))
        seed_groups: dict = defaultdict(list)
        for r in alpha0_runs:
            seed_groups[(r["cell"], r["algorithm"], r["behaviour"])].append(r)
        seed_stds = []
        for g in seed_groups.values():
            vals = [rr["eval_return_mean"] for rr in g if rr["eval_return_mean"] is not None]
            if len(vals) > 1:
                seed_stds.append(float(np.std(vals)))
        max_spread = max(spreads) if spreads else 0.0
        typical_seed_std = float(np.mean(seed_stds)) if seed_stds else 1.0
    c3_verdict = (
        "not_yet_testable" if max_spread is None
        else "supported" if max_spread < 2.0 * (typical_seed_std or 1.0)
        else "ambiguous"
    )

    r2_cv = regression.get("r2_pooled_cv")
    if r2_cv is None:
        c4_verdict = "not_yet_testable"
    elif r2_cv > 0.7:
        c4_verdict = "supported (above upper end)"
    elif r2_cv >= 0.4:
        c4_verdict = "supported"
    else:
        c4_verdict = "refuted"

    n_non_id = regression.get("n_non_id", 0) or 0
    c5_verdict = "supported" if n_non_id > 50 else "not_yet_testable"

    return {
        "claim_1_id_separation": {
            "expected": "id-cells gap ≪ partial_id-cells gap",
            "observed_id_mean_gap": id_mean_gap,
            "observed_partial_id_mean_gap": pid_mean_gap,
            "delta": delta_gap,
            "verdict": c1_verdict,
        },
        "claim_2_dtv_separates_strata": {
            "expected": "delta_tv larger in partial_id than in id",
            "observed_id_dtv": id_dtv,
            "observed_partial_id_dtv": pid_dtv,
            "ratio": ratio,
            "verdict": c2_verdict,
        },
        "claim_3_behaviour_silent_at_alpha_zero": {
            "expected": "no measurable behaviour-policy effect at alpha_conf=0",
            "observed_max_within_cell_spread": max_spread,
            "observed_seed_std_typical": typical_seed_std,
            "verdict": c3_verdict,
        },
        "claim_4_pooled_r2_in_range": {
            "expected": "pooled R^2 in [0.4, 0.7]",
            "observed": r2_cv,
            "verdict": c4_verdict,
        },
        "claim_5_non_id_populated": {
            "expected": "non_id stratum has n > 50 runs",
            "observed_n": n_non_id,
            "verdict": c5_verdict,
        },
    }


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------


def make_summary(results_dir: Path, output_path: Path | None = None) -> Path:
    """Aggregate *results_dir* into a ``summary.json`` file.

    Parameters
    ----------
    results_dir:
        Directory produced by ``run_full_matrix.py`` (contains per-run subdirs
        each with ``eval.csv``, ``train.csv``, ``eval_perturbed.csv``,
        ``meta.json``).
    output_path:
        Where to write the JSON.  Defaults to ``results_dir/summary.json``.

    Returns
    -------
    Path to the written file.
    """
    if output_path is None:
        output_path = results_dir / "summary.json"

    runs = _collect_runs(results_dir)

    # ---- Config metadata ----
    manifest_path = results_dir / "manifest.csv"
    n_manifest = 0
    seeds: list[int] = []
    alpha_sweep: list[float] = []
    if manifest_path.exists():
        manifest_rows = _read_all_rows(manifest_path)
        n_manifest = len(manifest_rows)
        seeds = sorted({int(r["seed"]) for r in manifest_rows})
        alpha_sweep = sorted({float(r["alpha_conf"]) for r in manifest_rows})

    n_envs: int | None = None
    total_frames: int | None = None
    for meta_path in results_dir.rglob("meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_envs = meta.get("n_envs")
        total_frames = meta.get("total_frames")
        break

    # ---- Wall time ----
    min_wt: float = float("inf")
    max_wt: float = float("-inf")
    for train_path in results_dir.rglob("train.csv"):
        all_rows = _read_all_rows(train_path)
        if not all_rows:
            continue
        wt_start = _sf(all_rows[0].get("wall_time"))
        wt_end = _sf(all_rows[-1].get("wall_time"))
        if wt_start is not None:
            min_wt = min(min_wt, wt_start)
        if wt_end is not None:
            max_wt = max(max_wt, wt_end)
    wall_time: float | None = (
        max_wt - min_wt if math.isfinite(min_wt) and math.isfinite(max_wt) else None
    )

    # ---- Groupings ----
    cells = sorted({r["cell"] for r in runs})
    algos = sorted({r["algorithm"] for r in runs})

    per_cell: dict[str, Any] = {}
    for cell in cells:
        group = [r for r in runs if r["cell"] == cell]
        s = _group_summary(group)
        s["shorthand"] = _CELL_SHORTHANDS.get(cell, f"cell_{cell}")
        per_cell[str(cell)] = s

    per_algo: dict[str, Any] = {
        algo: _group_summary([r for r in runs if r["algorithm"] == algo]) for algo in algos
    }

    per_cell_per_algo: dict[str, Any] = {}
    for cell in cells:
        for algo in algos:
            group = [r for r in runs if r["cell"] == cell and r["algorithm"] == algo]
            if group:
                per_cell_per_algo[f"{cell}_{algo}"] = _group_summary(group)

    per_cell_per_env: dict[str, Any] = {}
    for cell in cells:
        for env in _ALL_ENVS:
            group = [r for r in runs if r["cell"] == cell and r["env"] == env]
            per_cell_per_env[f"{cell}_{env}"] = (
                _group_summary(group) if group else {"eval_return_mean": None}
            )

    # Per-horizon breakdown (only emitted when the manifest contains multiple
    # horizons, i.e. a horizon_sweep was active).
    horizons = sorted({r["horizon"] for r in runs if r.get("horizon") is not None})
    per_horizon_summary: dict[str, Any] | None = None
    per_cell_per_horizon: dict[str, Any] | None = None
    if len(horizons) > 1:
        per_horizon_summary = {
            str(h): _group_summary([r for r in runs if r.get("horizon") == h])
            for h in horizons
        }
        per_cell_per_horizon = {}
        for cell in cells:
            for h in horizons:
                group = [
                    r for r in runs if r["cell"] == cell and r.get("horizon") == h
                ]
                if group:
                    per_cell_per_horizon[f"{cell}_h{h}"] = _group_summary(group)

    # ---- Regression + claims ----
    regression = _compute_regression(results_dir)
    claims = _compute_claims(runs, regression)

    bias_strengths = sorted(
        {r["bias_strength"] for r in runs if r["bias_strength"] is not None}
    )

    summary: dict[str, Any] = {
        "run_id": results_dir.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_runs_in_manifest": n_manifest,
            "n_envs": n_envs,
            "total_frames": total_frames,
            "seeds": seeds,
            "alpha_conf_sweep": alpha_sweep,
            "bias_strengths": bias_strengths,
        },
        "n_runs": len(runs),
        "n_runs_failed": max(0, n_manifest - len(runs)),
        "wall_time_seconds": wall_time,
        "git_sha": _git_sha(),
        "per_cell_summary": per_cell,
        "per_algorithm_summary": per_algo,
        "per_cell_per_algorithm": per_cell_per_algo,
        "per_cell_per_env": per_cell_per_env,
        "headline_regression": regression,
        "claims_evidence": claims,
    }
    if per_horizon_summary is not None:
        summary["per_horizon_summary"] = per_horizon_summary
        summary["per_cell_per_horizon"] = per_cell_per_horizon

    output_path.write_text(
        json.dumps(_nan_to_null(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate a results directory into summary.json."
    )
    parser.add_argument("--results", required=True, type=Path, help="Results directory")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output path (default: <results>/summary.json)"
    )
    args = parser.parse_args()
    out = make_summary(args.results, args.output)
    print(f"Summary written to: {out}")


if __name__ == "__main__":
    main()
