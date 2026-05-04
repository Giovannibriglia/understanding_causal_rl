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

# Always-keep columns: the stratification axis must be available even if it
# happens to be collinear with another feature, because the stratified-R²
# block reads from it.
_NEVER_DROP = {"id_status_ordinal"}

_ID_STATUS_COLORS = {"id": "#4CAF50", "partial_id": "#FF9800", "non_id": "#F44336"}


def _safe_float(val: object, default: float = 0.0) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _row_from_eval(
    last_eval: dict[str, str],
    target_G: float,
    *,
    D_env_KS: float | None = None,
    D_bisim: float | None = None,
    eps_T: float = 0.0,
    eps_R: float = 0.0,
) -> dict[str, Any]:
    id_status = str(last_eval.get("id_status", "non_id"))
    return {
        "G": target_G,
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
        "pi_b_recovery_kl": _safe_float(
            last_eval.get("pi_b_recovery_kl"), default=float("nan")
        ),
        "backdoor_residual_mean": _safe_float(
            last_eval.get("backdoor_residual_mean"), default=float("nan")
        ),
        "target_support_overlap": _safe_float(
            last_eval.get("target_support_overlap"), default=float("nan")
        ),
        "cond_mi_r_z_given_sa": _safe_float(
            last_eval.get("cond_mi_r_z_given_sa"), default=float("nan")
        ),
        "bias_strength": _safe_float(last_eval.get("bias_strength"), default=1.0),
        "D_env_KS": (
            float(D_env_KS) if D_env_KS is not None else float("nan")
        ),
        "D_bisim": float(D_bisim) if D_bisim is not None else float("nan"),
        "alpha_conf": _safe_float(last_eval.get("alpha_conf")),
        "expose_z": _safe_float(last_eval.get("expose_z")),
        "pi_b_known": _safe_float(last_eval.get("pi_b_known")),
        "eps_T": float(eps_T),
        "eps_R": float(eps_R),
    }


def _load_joined_data(
    results_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Join eval.csv with eval_perturbed.csv when available; otherwise fall
    back to in-domain mode (G = oracle - eval_return).

    Returns ``(rows, mode)`` where ``mode`` is one of
    ``"generalisation"``, ``"in_domain"`` (only ``eval.csv`` available), or
    ``"empty"`` (no usable rows).
    """
    rows: list[dict[str, Any]] = []
    n_with_perturbed = 0
    n_without_perturbed = 0
    for eval_path in results_dir.rglob("eval.csv"):
        run_dir = eval_path.parent
        with eval_path.open("r", encoding="utf-8") as f:
            eval_rows = list(csv.DictReader(f))
        if not eval_rows:
            continue
        last_eval = eval_rows[-1]
        oracle_return = _safe_float(last_eval.get("eval_oracle_return_mean"))
        eval_return = _safe_float(last_eval.get("eval_return_mean"))

        perturbed_path = run_dir / "eval_perturbed.csv"
        perturbed_rows: list[dict[str, str]] = []
        if perturbed_path.exists():
            with perturbed_path.open("r", encoding="utf-8") as f:
                perturbed_rows = list(csv.DictReader(f))

        if perturbed_rows:
            n_with_perturbed += 1
            for pr in perturbed_rows:
                perturbed_return = _safe_float(pr.get("eval_perturbed_return_mean"))
                rows.append(
                    _row_from_eval(
                        last_eval,
                        target_G=oracle_return - perturbed_return,
                        D_env_KS=_safe_float(pr.get("D_env_KS")),
                        D_bisim=_safe_float(pr.get("D_bisim"), default=float("nan")),
                        eps_T=_safe_float(pr.get("eps_T")),
                        eps_R=_safe_float(pr.get("eps_R")),
                    )
                )
        else:
            n_without_perturbed += 1
            # In-domain fallback: G = within-train return gap.
            rows.append(
                _row_from_eval(
                    last_eval,
                    target_G=oracle_return - eval_return,
                )
            )
    if not rows:
        return rows, "empty"
    # Mode = "generalisation" iff every joined row came from a perturbed CSV.
    if n_with_perturbed > 0 and n_without_perturbed == 0:
        return rows, "generalisation"
    if n_without_perturbed > 0 and n_with_perturbed == 0:
        return rows, "in_domain"
    # Mixed: prefer the larger source.
    return rows, ("generalisation" if n_with_perturbed >= n_without_perturbed else "in_domain")


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


def _is_constant(column: np.ndarray[Any, Any]) -> bool:
    """Constant means strictly identical (up to NaN) across samples."""
    finite = column[~np.isnan(column)] if column.dtype.kind == "f" else column
    if finite.size == 0:
        return True
    return bool(np.unique(finite).size <= 1)


def _partial_r2(
    X: np.ndarray[Any, Any],  # noqa: N803
    y: np.ndarray[Any, Any],
    feature_idx: int,
    estimator_factory: Any,
) -> float:
    """R² gain attributable to a single feature.

    Equals ``R²(X) - R²(X without feature_idx)``.  Clamped to ``[0, 1]``.
    """
    full = estimator_factory().fit(X, y).score(X, y)
    if X.shape[1] <= 1:
        return max(0.0, min(1.0, full))
    X_drop = np.delete(X, feature_idx, axis=1)  # noqa: N806
    drop = estimator_factory().fit(X_drop, y).score(X_drop, y)
    return float(max(0.0, full - drop))


def make_headline_regression(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Fit and plot the headline regression.

    Two-mode operation:

    * **Generalisation mode** when ``eval_perturbed.csv`` is available — target
      is ``G = oracle_return - perturbed_return``.
    * **In-domain mode** when only ``eval.csv`` is available — target is
      ``G = oracle_return - eval_return``.

    Generates ``headline_regression_table.csv``, ``headline_stratified_r2.csv``,
    ``headline_pure_divergence.pdf``, ``headline_fitted.pdf`` and a
    ``headline_regression_meta.json`` with the run mode.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]
        from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
        from sklearn.linear_model import RidgeCV  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for headline_regression. "
            "Install it with: pip install scikit-learn"
        ) from e

    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, mode = _load_joined_data(results_dir)
    print(f"[headline_regression] mode = {mode}, n_rows = {len(rows)}")
    if len(rows) < 10:
        return

    # Build feature matrix
    feature_cols = [c for c in _FEATURE_COLS if c in rows[0]]
    X_raw = np.array(  # noqa: N806
        [[r.get(c, float("nan")) for c in feature_cols] for r in rows], dtype=np.float64
    )
    y = np.array([r["G"] for r in rows], dtype=np.float64)
    id_statuses = [str(r["id_status"]) for r in rows]

    # NaN imputation: replace per-column NaNs with that column's median
    # (or 0 when entirely NaN).  This is gentler than zeroing everything
    # and keeps the surviving signal usable for regression.
    for j in range(X_raw.shape[1]):
        col = X_raw[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            X_raw[:, j] = 0.0
            continue
        if nan_mask.any():
            X_raw[nan_mask, j] = float(np.nanmedian(col))

    # Drop *truly* constant features (single unique value).  Variable but
    # tiny-variance features are kept so RidgeCV can shrink them on its own.
    keep_mask = np.array(
        [
            (feature_cols[i] in _NEVER_DROP) or not _is_constant(X_raw[:, i])
            for i in range(len(feature_cols))
        ]
    )
    dropped_constant_features = [
        feature_cols[i] for i in range(len(feature_cols)) if not keep_mask[i]
    ]
    feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if keep_mask[i]]
    X_raw = X_raw[:, keep_mask]

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)  # noqa: N806

    # Use RidgeCV: handles collinear features gracefully via L2.
    def _make_lin() -> Any:
        return RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0))

    lin = _make_lin()
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

    # Per-feature partial R² (gain over leave-one-out baseline).
    partial_r2 = np.zeros(len(feature_cols))
    for i in range(len(feature_cols)):
        try:
            partial_r2[i] = _partial_r2(X, y, i, _make_lin)
        except Exception:  # noqa: BLE001
            partial_r2[i] = float("nan")

    # Save regression table (includes dropped constant features with coef=NaN).
    table_path = output_dir / "headline_regression_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["feature", "coef", "ci_lo", "ci_hi", "importance_rf", "partial_r2", "dropped"]
        )
        for i, feat in enumerate(feature_cols):
            writer.writerow(
                [
                    feat,
                    lin.coef_[i],
                    ci_lo[i],
                    ci_hi[i],
                    rf_importances[i],
                    float(partial_r2[i]),
                    False,
                ]
            )
        for feat in dropped_constant_features:
            writer.writerow(
                [
                    feat,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    True,
                ]
            )

    # Save mode metadata.
    import json as _json

    meta_path = output_dir / "headline_regression_meta.json"
    meta_path.write_text(
        _json.dumps(
            {
                "mode": mode,
                "n_rows": len(rows),
                "n_features_kept": len(feature_cols),
                "n_features_dropped": len(dropped_constant_features),
                "dropped_features": dropped_constant_features,
                "r2_pooled_train": r2_train,
                "r2_pooled_cv": r2_cv,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

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
