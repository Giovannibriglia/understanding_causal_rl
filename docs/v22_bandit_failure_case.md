# v22 — bandit failure case (paper_bandit_offpolicy.yaml)

## The paper claim this run supports

The diagnostic taxonomy correctly distinguishes solvable from
unsolvable cases on bandits.  On *the same buffer* of confounded
behaviour-policy data, an IPW-corrected per-arm estimator recovers
the optimal arm and a naive estimator does not — and the diagnostic
`Δ_TV` reflects the difference.  This is the controlled-pair
contrast the bandit section of the paper leans on.

## Why a separate config from `paper_bandit.yaml`

| Config | Behaviour | Algos | What it shows |
|--------|-----------|-------|---------------|
| `paper_bandit.yaml` | uniform | UCB, UCB±, UCB+, RCT | Success case: on-policy algos find near-optimal policies in all 4 cells; the `expose_z` axis is the only meaningful separator (`pi_b_known` collapses for on-policy). |
| `paper_bandit_offpolicy.yaml` (this PR) | uniform + reward_aligned | offline_bandit_naive, offline_bandit_ipw | Failure case: under confounded behaviour data, naive estimation fails and IPW recovers.  Both algos see the same buffer. |

The paper's bandit section will draw on both.  Combining the two
runs' `summary.json` files is the paper-writing step; the runner
treats them as separate matrix runs.

## Per-cell expected signature

The empirical signature mirrors the YAML header table:

* **Cell 1 (`expose_z=T, pi_b_known=T`)**:
  * `reward_aligned × α_conf=4`: naive's `gap_to_oracle` is large;
    IPW with the *true* behaviour log-probs is small.  This is the
    headline failure-case contrast.
  * `uniform` (or `α_conf=0`): both algos produce similar gaps —
    no confounding means no IPW correction is needed.
* **Cell 2 (`expose_z=T, pi_b_known=F`)**:
  * IPW relies on the propensity-model estimate of `π_b`, not the
    true logprob.  Fitting error introduces a small bias; gap should
    sit between Cell 1 IPW (true logprobs, smallest gap) and naive
    (no IPW, largest gap).
* **Cell 3 (`expose_z=F, pi_b_known=T`)**:
  * The marginalised oracle is weaker than C1's per-Z oracle (v19's
    fix).  IPW correction helps with confounding but doesn't recover
    the per-Z information the agent can't see.  Both algos have
    smaller `gap_to_oracle` than C1 because the oracle is weaker;
    the *relative* IPW-vs-naive contrast remains.
* **Cell 4 (`expose_z=F, pi_b_known=F`)**:
  * Both errors compound (marginalised oracle + propensity-model
    estimate).  Largest gap among the four cells; the IPW-vs-naive
    contrast is dampened by the propensity estimation error.

## Reading the resulting figures

* **`bandit_regret.pdf`** (post-v20 layout: C1 vs C3 facets, one line
  per algo): the four lines from `paper_bandit.yaml` (UCB-family +
  RCT) should match their pre-v22 numbers when re-rendered against
  the same data.  The two new lines (naive, IPW) should differ
  visibly under `reward_aligned × α_conf=4`: naive flat or rising
  (regret accumulates because the agent sticks with a wrong arm),
  IPW curving down toward zero.
* **`headline_regression.pdf`**: the regression should now find a
  stronger relationship between `Δ_TV` and `gap_to_oracle` on the
  combined dataset, because the algorithm-axis variation (IPW vs
  naive on the same buffer) creates real gap variation that `Δ_TV`
  can predict.  Pre-v22 the bandit family had pooled R²=0.227 with
  `id_status_ordinal` doing all the work; post-v22 the IPW-vs-naive
  split should contribute meaningful within-cell variation.
* **`headline_pure_divergence.pdf`** (post-v20 fallback to
  `Δ_TV`-alone): with `eval_perturbations=false` the figure plots
  `Δ_TV` alone.  The IPW-vs-naive separation should be visible as
  two clusters at large `α_conf` even within the same id_status
  stratum.

## What to expect about `claims_evidence`

Pre-v22, `claim_4` (pooled R² ∈ [0.4, 0.7]) was likely *not*
supported by `paper_bandit` alone — the on-policy success case
collapses the within-cell variation that the regression needs.
Combining with `paper_bandit_offpolicy` should push pooled R² into
or near the [0.4, 0.7] band; the deciding factor is whether
`id_status_ordinal` still dominates partial R² or whether `Δ_TV`
gains within-cell predictive power.

The paper's headline number will be reported on the combined
dataset; this PR doesn't perform the combination (deferred to the
paper-writing step).

## Out of scope

* Running the full 320-run matrix.  User's compute step.
* Combining `paper_bandit` and `paper_bandit_offpolicy` summaries
  into a single `summary.json`.  Paper-writing step.
* New algos, new behaviour-policy classes, env changes.  v21 shipped
  the algos; v22 only adds a config and this docs note.
