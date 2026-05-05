# Figures reference

The repo's figures are organised as a *diagnostic walkthrough*, not as a
prediction pipeline.  Read top-to-bottom: regime map → coverage signature
→ flowchart → per-cell trajectories → robustness checks → supplementary
regression.  The reader's goal is to find their own run on the regime
map, then walk down the flowchart to a one-sentence diagnosis.

## Section 4 — Headline diagnosis figures (v11)

1. **`regime_map.pdf`** — four-quadrant scatter of (Δ_TV, gap_to_oracle)
   with overlaid quadrant labels.  Each run is one point.  Replaces
   ``headline_gap_vs_oracle.png`` as the primary figure.
2. **`diagnosis_flowchart.pdf`** — static decision tree from
   ``gap_to_oracle`` through Δ_TV, ``ess_ratio``, and ``expose_z`` to one
   of four diagnoses.  The practitioner-facing walkthrough.
3. **`coverage_signature.pdf`** — two-panel (ESS, min_propensity) scatter
   that exposes the coverage-broken regime when ``coverage_stress`` data
   is present.  Without it, one cluster; with it, two visibly different
   clusters.

## Section 4 — Per-cell diagnostics

4. **`metrics_curves_cell{1..4}.png`** — six Δ-divergence trajectories
   per cell (TV, KL, χ², sup, MMD², KS).  Comprehensive view of all
   evolving divergences.  Pre-v13 this figure plotted offline-buffer
   properties that don't change during training; those moved to
   ``static_diagnostics_{max,std}.pdf`` (figure 7 below) where a box
   plot is the right idiom.
5. **`gap_curves_cell{1..4}.png`** — divergence-family time-series per
   cell.  Δ_TV / Δ_KL / Δ_χ² / Δ_sup over training.  The narrative
   4-panel; ``metrics_curves`` is the comprehensive 6-panel.

## Section 4 — Supplementary diagnostics

7. **`static_diagnostics_max.pdf`** and **`static_diagnostics_std.pdf`** —
   two variants of the static offline-buffer diagnostics figure, one
   box per cell.  Five panels are shared (``min_propensity``,
   ``ess_ratio``, ``coverage_ess_histogram``,
   ``coverage_action_freq_min``, ``propensity_calibration_ece``); the
   sixth panel differs:

   - ``_max``: ``bound_width_max`` — width of the rarest-played arm.
     Grows with π_b bias because rare arms have large ``1 − p_a``.
   - ``_std``: ``bound_width_std`` — population std across arms.  Zero
     under uniform π_b (all per-arm widths equal ``1 − 1/n_actions``);
     grows monotonically with bias.

   ``bound_width_mean`` is **not** plotted because it is mathematically
   constant at ``1 − 1/n_actions = 0.875`` for any ``p_a`` summing to 1
   — an algebraic identity, not a measurement.  ``max`` and ``std`` are
   the two informative aggregates; the test
   ``tests/metrics/test_bound_width_aggregates.py`` codifies the
   contract.

   These metrics describe the offline dataset, not the trained policy,
   so they're the same value at every checkpoint within a run — the
   cross-run variation only shows up across runs (one box per cell) or
   across regimes.  Use this when reading the dataset's coverage /
   identifiability properties; use ``metrics_curves`` and ``gap_curves``
   when reading the trained policy's behaviour.

## Section 4 — Robustness and stratification

6. **`divergence_panel.pdf`** — five divergences (TV, KL, χ², sup, MMD²,
   KS) at the final checkpoint, stratified by id_status.  Sanity check
   that the qualitative pattern is robust to the choice of divergence.
7. **`causal_metrics_grid.pdf`** — three panels for the v8 causal-RL
   metrics (backdoor residual, conditional MI, target-support overlap)
   stratified by cell.
8. **`identifiability_panel.pdf`** — return gap to oracle by id_status.

## Section 4 — Bias and coverage sweeps

9. **`bias_sweep.pdf`** — Δ_TV / ESS / min_propensity / gap vs
   ``bias_strength``, stratified by id_status.
10. **`sample_sweep.pdf`** — final gap vs offline-data size; renders
    only when the manifest contains ≥ 3 distinct ``offline_transitions``.
11. **`bound_width_panel.pdf`** — per-arm bound width vs per-arm
    sub-optimality.  Caption notes the structural pinning at
    ``1 − 1/n_actions`` under uniform π_b.

## Section 4 — Bandit family

12. **`bandit_regret.pdf`** — UCB / UCB- / UCB+ / RCT cumulative regret.

## Section 5 — Supplementary (regression)

The regression-framed figures are demoted from headline to supplementary:
some reviewers will want a regression score, but the diagnostic story is
told without it.

13. `headline_pure_divergence.pdf` — G vs divergence summary.
14. `headline_fitted.pdf` — G vs fitted linear-model prediction.
15. `headline_gap_vs_oracle.png` — final gap vs Δ_TV scatter (legacy).

The headline-regression CSVs (`headline_regression_table.csv`,
`headline_stratified_r2.csv`, `headline_regression_meta.json`) are still
produced by ``make_headline_regression``.  Use them when reporting
quantitative R²; lead with the diagnostic figures otherwise.
