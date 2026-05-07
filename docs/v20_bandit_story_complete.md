# v20 — bandit story complete

v19 fixed the bandit `do_reward` and the analytical oracle, but a
`paper_bandit --quick` smoke run on the v19 branch surfaced a
preexisting `BanditView` alignment bug and two figure-rendering
problems.  v20 fixes all three so the bandit-side end-to-end story is
coherent.

## Phase 1 — `BanditView.reset` state-action-reward alignment

Pre-v20, `BanditView.reset` permuted the inner env's obs pool via
`obs[idx]` where `idx` was a random sample of size `n_envs`.  But
`BanditView.step` calls `self._env.do_reward(action)`, which indexes
the inner env's *un-permuted* `_latent_state` directly.  So the agent
at env position `i` saw an obs corresponding to state
`_latent_state[idx[i]]` while the reward at the same position was
computed against `_latent_state[i]` — a different state.

The v19 analytical oracle reads `inner._latent_state[i]` to pick the
optimal action; the agent's policy was trained on (and acted from) obs
corresponding to a different state, so its actions were systematically
suboptimal w.r.t. the env's reward computation.  Empirically this
showed up as `gap_to_oracle = -0.25` for every UCB-family algorithm
in C3 on the v19 quick smoke (a sign-flip of the gap because the
"agent" and "oracle" labels are about *which* state's action they
chose, and both end up scoring against the same un-permuted state).

The permuted `_context_pool` attribute set in `__init__` was unused
downstream (`grep -rn _context_pool src/ tests/` found only the
assignment site), so dropping the permutation simply removed dead
code that incidentally created the alignment bug.

## Phase 2 — `bandit_regret.pdf` rewrite

The previous figure had four bugs: identical Δ panels (no filtering),
across-runs cumsum (broke within-run monotonicity), a derived "RCT
outperforms UCB" story that was a consequence of the cumsum bug, and
a fictional Δ axis the env doesn't actually parameterise on.

The rewrite plots cumulative expected regret per training step,
faceted by cell, with mean ± standard error across seeds.  v19's
cell collapse means C1 ≡ C2 and C3 ≡ C4 in expectation, so two
facets {C1, C3} are sufficient — they capture the meaningful
`expose_z` axis without redundancy.  The within-run cumsum is
factored out as `_within_run_cum_regret` so the contract is
unit-testable.

## Phase 3 — `headline_pure_divergence.pdf` fallback

Pre-v20 the figure required *both* `delta_tv` and `D_env_KS` to be in
`feature_cols`.  `D_env_KS` is dropped as a constant feature whenever
`eval_perturbations=false` (every bandit run, the v17 `smoke_v8`
budget cuts, and any config that skips perturbation eval).  When that
happened, the figure rendered as empty axes with the "Pure divergence
predictor" title.

v20 falls back to whichever divergence column is available; only
skips the figure entirely (with a log line) when neither is present.
The column-selection logic is factored as `_select_divergence_axis`
for direct unit testing.

## What v20 does *not* fix

* Adding a Δ (arm-gap) parameter to the bandit env.  The figure
  rewrite drops the fictional Δ axis; if a future paper draft wants
  a Δ-sweep, that's an env redesign, not a figure fix.
* Off-policy bandit algorithms (still v21+ if needed).
* `paper_bandit.yaml` further changes; v19 already updated it.
