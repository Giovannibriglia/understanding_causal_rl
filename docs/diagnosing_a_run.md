# Diagnosing a failing RL run

You trained an agent.  The eval return is below what you expected.  What
now?

This page is a practitioner's one-pager — a debugger's checklist that
walks from "my run looks bad" to one of four diagnoses, using diagnostic
metrics that are already in every run's CSV.

## The 7-step walkthrough

1. **Open `eval.csv` for the final checkpoint.**
2. **Note `eval_return_mean` and `eval_oracle_return_mean`.**  Compute
   `gap = oracle - eval`.
3. **If `gap < 5`, you're done.**  Whatever's left is sample complexity
   (or your environment is genuinely hard).  Stop here.

If `gap >= 5`, continue:

4. **Check `delta_tv` (final).**  If `delta_tv < 0.03`, the gap is *not*
   driven by an identifiability issue — it's coverage or sample
   complexity.  Skip to step 7.
5. **Check whether `expose_z = True`.**  If False, you have partial
   observability.  Stop and expose more state.
6. **Check `propensity_calibration_ece`.**  If > 0.05, your propensity
   model is mis-specified.  Refit.
7. **Check `ess_ratio` and `min_propensity`.**
   - `ess_ratio < 0.5`: the dataset is too small or too biased.  Collect
     more, or with higher action diversity.
   - `min_propensity < 0.01`: some action-state pairs are not in your
     buffer at all.  Re-collect with action-mask removed, or use a
     stochastic behaviour policy.

If you're at step 7 with healthy values, the residual gap is sample
complexity.  More samples or a tighter algorithm.

## The four regimes

The same logic can be read off the [regime map](figures.md#section-4--headline-diagnosis-figures-v11) figure.  Plot every run in
the `(delta_tv, gap)` plane and find the quadrant.  Each quadrant has a
one-sentence diagnosis:

| Quadrant | Δ_TV | gap | Diagnosis |
|---|---|---|---|
| Identifiable | low | low | "Sample complexity.  Done unless you want a tighter algorithm." |
| Coverage-broken | low | high | "ESS / min_propensity are low.  Collect data with higher action diversity." |
| Confounding visible, recoverable | high | low | "Algorithm sees the confounding signal.  Fit propensity, use IPW or CQL." |
| Partial observability | high | high | "Observed covariates are insufficient.  Expose more state." |

## See also

- `regime_map.pdf` — the four-quadrant figure that places every run on
  the diagnostic plane.
- `diagnosis_flowchart.pdf` — the static decision tree this page is built
  around.
- `coverage_signature.pdf` — `(ESS, min_propensity)` log-log scatter,
  shows whether your run sits in the healthy cluster or the
  coverage-broken cluster.
- `metrics_curves_cell{1..4}.png` — per-cell time-series of every
  diagnostic metric over training.
- `causal_metrics_grid.pdf` — three boxplots for the v8 causal-RL
  metrics (backdoor residual, conditional MI, target-support overlap)
  stratified by cell.

## Reproducing the coverage-broken quadrant

If your dataset is naturally healthy and you want to validate the
diagnosis flow, run

```bash
python scripts/run_full_matrix.py --config coverage_stress --n-workers 4
python scripts/make_all_figures.py --results results/coverage_stress_<timestamp>
```

The `masked_4` regime (forbidden_actions = `[0, 1, 2, 3]`) deliberately
breaks coverage on half the action space — the resulting `min_propensity`
falls below 0.001 on the forbidden arms and `ess_ratio` below 0.4.  Cell-1
runs in this regime show return gaps comparable to partial-id cells.
This is the cleanest empirical demonstration that coverage is its own
failure mode, distinct from graphical identifiability.
