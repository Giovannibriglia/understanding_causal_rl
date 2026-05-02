"""Headline regression: oracle gap G ≈ f(delta_tv, bound_width, coverage, D_env, ...).

PRE-COMMITTED CLAIM (asserted in tests):
  Across cells, the oracle return gap G is predicted by a sparse combination
  of (a) within-env identifiability gap delta_tv, (b) natural-bound width when
  applicable, (c) coverage diagnostics (ess_ratio, overlap), and (d) the
  environment-level shift D_env_KS.

  We expect:
    R²(linear, all cells pooled) ∈ [0.4, 0.7]
    R²(linear, id-stratum)        ∈ [0.6, 0.9]   — divergence dominates
    R²(linear, partial_id)        ∈ [0.5, 0.8]   — divergence + bound width
    R²(linear, non_id)            ∈ [0.0, 0.3]   — non-identifiability cannot
                                                   be summarised by a divergence

  A low R² in the non_id stratum is the *result*, not a failure.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from causal_rl.identification.id_oracle import ID_STATUS_ORDER
from causal_rl.plotting.style import apply_style

_FEATURE_COLS = [
    "delta_tv",
    "bound_width_mean",
    "min_propensity",
    "ess_ratio",
    "overlap_at_1e-2",
    "tail_mass_top10pct",
    "propensity_calibration_ece",
    "bias_strength",
    "id_status_ordinal",
    "D_env_KS",
    "alpha_conf",
    "expose_z",
    "pi_b_known",
]

_ID_STATUS_COLORS = {"id": "#4CAF50", "partial_id": "#FF9800", "non_id": "#F44336"}


def _safe_float(val: object, default: float = 0.0) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_joined_data(
    results_dir: Path,
) -> list[dict[str, Any]]:
    """Join eval.csv, train.csv, and eval_perturbed.csv on run identity."""
    rows: list[dict[str, Any]] = []
    for eval_path in results_dir.rglob("eval.csv"):
        run_dir = eval_path.parent
        with eval_path.open("r", encoding="utf-8") as f:
            eval_rows = list(csv.DictReader(f))
        if not eval_rows:
            continue
        last_eval = eval_rows[-1]

        # Try to load perturbed eval
        perturbed_path = run_dir / "eval_perturbed.csv"
        if not perturbed_path.exists():
            continue
        with perturbed_path.open("r", encoding="utf-8") as f:
            perturbed_rows = list(csv.DictReader(f))
        if not perturbed_rows:
            continue

        oracle_return = _safe_float(last_eval.get("eval_oracle_return_mean"))
        for pr in perturbed_rows:
            perturbed_return = _safe_float(pr.get("eval_perturbed_return_mean"))
            G = oracle_return - perturbed_return  # noqa: N806
            id_status = str(last_eval.get("id_status", "non_id"))
            row: dict[str, Any] = {
                "G": G,
                "id_status": id_status,
                "id_status_ordinal": ID_STATUS_ORDER.get(id_status, 2),
                "delta_tv": _safe_float(last_eval.get("delta_tv")),
                "bound_width_mean": _safe_float(last_eval.get("bound_width_mean")),
                "min_propensity": _safe_float(last_eval.get("min_propensity")),
                "ess_ratio": _safe_float(last_eval.get("ess_ratio")),
                "overlap_at_1e-2": _safe_float(last_eval.get("overlap_at_1e-2")),
                "tail_mass_top10pct": _safe_float(last_eval.get("tail_mass_top10pct")),
                "propensity_calibration_ece": _safe_float(
                    last_eval.get("propensity_calibration_ece"), default=float("nan")
                ),
                "bias_strength": _safe_float(last_eval.get("bias_strength"), default=1.0),
                "D_env_KS": _safe_float(pr.get("D_env_KS")),
                "alpha_conf": _safe_float(last_eval.get("alpha_conf")),
                "expose_z": _safe_float(last_eval.get("expose_z")),
                "pi_b_known": _safe_float(last_eval.get("pi_b_known")),
                "eps_T": _safe_float(pr.get("eps_T")),
                "eps_R": _safe_float(pr.get("eps_R")),
            }
            rows.append(row)
    return rows


_MIN_SAMPLES_CV = 6
_MIN_SAMPLES_STRATUM = 10


def _r2_score(y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _safe_cv_r2(
    X: np.ndarray[Any, Any],  # noqa: N803
    y: np.ndarray[Any, Any],
    n_folds: int = 5,
) -> float:
    """Cross-validated R² with guard for small n. Returns NaN when n < _MIN_SAMPLES_CV."""
    from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
    from sklearn.model_selection import cross_val_score  # type: ignore[import-untyped]

    n = len(y)
    if n < _MIN_SAMPLES_CV:
        return float("nan")
    cv = min(n_folds, n // 2)
    scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def make_headline_regression(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Fit and plot the headline regression.

    Requires scikit-learn. Generates:
      - headline_regression_table.csv
      - headline_stratified_r2.csv
      - headline_pure_divergence.pdf
      - headline_fitted.pdf
    """
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
        from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for headline_regression. "
            "Install it with: pip install scikit-learn"
        ) from e

    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_joined_data(results_dir)
    if len(rows) < 10:
        return

    # Build feature matrix
    feature_cols = [c for c in _FEATURE_COLS if c in rows[0]]
    X_raw = np.array(  # noqa: N806
        [[r.get(c, 0.0) for c in feature_cols] for r in rows], dtype=np.float64
    )
    y = np.array([r["G"] for r in rows], dtype=np.float64)
    id_statuses = [str(r["id_status"]) for r in rows]

    # Impute NaN propensity ECE with median
    ece_col = (
        feature_cols.index("propensity_calibration_ece")
        if "propensity_calibration_ece" in feature_cols
        else -1
    )
    if ece_col >= 0:
        mask = np.isnan(X_raw[:, ece_col])
        if mask.any() and (~mask).any():
            X_raw[mask, ece_col] = np.nanmedian(X_raw[:, ece_col])
        elif mask.all():
            X_raw[:, ece_col] = 0.0

    # Drop constant features (std < 1e-8) before scaling — they produce
    # meaningless zero coefficients and clutter the regression table.
    feature_stds = X_raw.std(axis=0)
    keep_mask = feature_stds >= 1e-8
    dropped_constant_features = [feature_cols[i] for i in range(len(feature_cols)) if not keep_mask[i]]
    feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if keep_mask[i]]
    X_raw = X_raw[:, keep_mask]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)  # noqa: N806

    # Pooled linear regression
    lin = LinearRegression()
    lin.fit(X, y)
    y_pred_lin = lin.predict(X)
    r2_train = _r2_score(y, y_pred_lin)
    r2_cv = _safe_cv_r2(X, y)

    # Random forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=0)
    rf.fit(X, y)
    rf_importances = rf.feature_importances_

    # Bootstrap CIs for linear coefficients
    n_boot = 200
    boot_coefs = np.zeros((n_boot, len(feature_cols)))
    rng = np.random.default_rng(42)
    n = len(rows)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lr_b = LinearRegression()
        lr_b.fit(X[idx], y[idx])
        boot_coefs[b] = lr_b.coef_

    ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
    ci_hi = np.percentile(boot_coefs, 97.5, axis=0)

    # Stratified R²: fit per-stratum linear model and report train + CV R²
    strat_r2_rows: list[dict[str, object]] = []
    for status in ["id", "partial_id", "non_id"]:
        s_mask = np.array([s == status for s in id_statuses])
        n_s = int(s_mask.sum())
        if n_s < _MIN_SAMPLES_STRATUM:
            strat_r2_rows.append(
                {"stratum": status, "r2_train": float("nan"), "r2_cv": float("nan"), "n": n_s}
            )
            continue
        lr_s = LinearRegression()
        lr_s.fit(X[s_mask], y[s_mask])
        r2_s_train = _r2_score(y[s_mask], lr_s.predict(X[s_mask]))
        r2_s_cv = _safe_cv_r2(X[s_mask], y[s_mask])
        strat_r2_rows.append(
            {"stratum": status, "r2_train": r2_s_train, "r2_cv": r2_s_cv, "n": n_s}
        )
    strat_r2_rows.append({"stratum": "pooled", "r2_train": r2_train, "r2_cv": r2_cv, "n": n})

    # Save regression table (includes dropped constant features with coef=NaN).
    table_path = output_dir / "headline_regression_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "coef", "ci_lo", "ci_hi", "importance_rf", "dropped"])
        for i, feat in enumerate(feature_cols):
            writer.writerow([feat, lin.coef_[i], ci_lo[i], ci_hi[i], rf_importances[i], False])
        for feat in dropped_constant_features:
            writer.writerow([feat, float("nan"), float("nan"), float("nan"), float("nan"), True])

    # Save stratified R² table
    strat_path = output_dir / "headline_stratified_r2.csv"
    with strat_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stratum", "r2_train", "r2_cv", "n"])
        writer.writeheader()
        writer.writerows(strat_r2_rows)

    # Figure 1: G vs delta_tv + D_env_KS (per stratum)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    div_col = feature_cols.index("delta_tv") if "delta_tv" in feature_cols else -1
    denv_col = feature_cols.index("D_env_KS") if "D_env_KS" in feature_cols else -1
    if div_col >= 0 and denv_col >= 0:
        combined_x = X_raw[:, div_col] + X_raw[:, denv_col]
        for status in ["id", "partial_id", "non_id"]:
            mask = np.array([s == status for s in id_statuses])
            if not mask.any():
                continue
            color = _ID_STATUS_COLORS.get(status, "gray")
            ax1.scatter(combined_x[mask], y[mask], color=color, alpha=0.5, s=20, label=status)
            if mask.sum() > 2:
                xm, ym = combined_x[mask], y[mask]
                m, b = np.polyfit(xm, ym, 1)
                xs = np.linspace(xm.min(), xm.max(), 50)
                ax1.plot(xs, m * xs + b, color=color, linewidth=1.5)

    ax1.set_xlabel("Δ_TV + D_env_KS")
    ax1.set_ylabel("G (oracle gap)")
    ax1.set_title(f"Pure divergence predictor  (overall R²={r2_cv:.2f})")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(output_dir / "headline_pure_divergence.pdf", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: G vs linear model prediction
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for status in ["id", "partial_id", "non_id"]:
        mask = np.array([s == status for s in id_statuses])
        if not mask.any():
            continue
        ax2.scatter(
            y_pred_lin[mask],
            y[mask],
            color=_ID_STATUS_COLORS.get(status, "gray"),
            alpha=0.5,
            s=20,
            label=status,
        )
    mn = min(y.min(), y_pred_lin.min())
    mx = max(y.max(), y_pred_lin.max())
    ax2.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    ax2.set_xlabel("Predicted G")
    ax2.set_ylabel("Observed G")
    ax2.set_title(f"Headline regression (train R²={r2_train:.2f}, CV R²={r2_cv:.2f})")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "headline_fitted.pdf", bbox_inches="tight")
    plt.close(fig2)
