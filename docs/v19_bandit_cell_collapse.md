# v19 — bandit cell taxonomy fixes + analytical oracle

The `paper_bandit_20260506_181801` matrix run completed cleanly
(640/640) but inspection showed the four-cell taxonomy had collapsed
to one cell in the bandit setting, with negative `gap_to_oracle`
values across the board.  v19 fixes the three confounding root
causes; the figure-rendering bugs (`bandit_regret.pdf` non-monotone
regret, identical Δ panels, `headline_pure_divergence.pdf` empty
rendering) are out of scope and tracked for v20.

## Phase 1 — `TabularSepsisEnv.do_reward` honours `expose_z`

Pre-v19, `do_reward(action)` returned
`reward_probs[self._latent_state, a]` unconditionally — conditioning
on the full latent state regardless of whether Z was exposed.
`BanditView.step` calls `do_reward` for every action, so an agent
in C3 (Z hidden) experienced the same reward distribution as C1
(Z observed).

v19 mirrors the `expose_z` branching that `sample_interventional`
has been honouring since the env was built: when Z is hidden,
marginalise uniformly over Z ∈ {0, 1} so the agent in C3/C4
experiences the harder marginalised distribution.

This change affects all callers of `do_reward` for C3/C4 envs (gap-
metric fallback paths in `metrics/runner_hooks.py` and
`metrics/plugin.py`, the `BanditView.step` reward-sampling path).
For C1/C2 the conditional branch is unchanged, so behaviour is
byte-identical.

## Phase 2 — bandit oracle uses the analytical DP optimum

Two related fixes to `BenchmarkRunner._oracle_action`:

* `BanditView` wraps `TabularSepsisEnv`, but the legacy
  `isinstance(eval_env, TabularSepsisEnv)` check missed the wrapper
  and bandit runs fell through to
  `algo.select_action(deterministic=True)`.  That made the "oracle"
  equal to the agent's own current best guess, producing nonsensical
  *negative* `gap_to_oracle` values (the agent could "beat" itself by
  RNG drift between training and eval calls).  v19 walks through
  wrapper layers (`getattr(env, "_env", env)`) until a
  `TabularSepsisEnv` is found, then runs the analytical DP oracle on
  that.
* `_build_tabular_oracle_policy` was reading
  `reward_probs[:, :, 1] - reward_probs[:, :, 0]` directly — the
  unmarginalised per-Z expectation.  In C3/C4 that gave the oracle
  access to information the agent cannot see, making it strictly
  stronger than achievable.  v19 marginalises over Z when
  `expose_z=False` so the oracle's reward function matches what the
  agent's environment actually presents.

### Caveat: this change affects the sequential env too

For the sequential `TabularSepsisEnv` (horizon=20), the C3/C4 oracle
in past matrices was conditioning on Z and giving an unattainable
upper bound.  Post-v19 the C3/C4 oracle is constrained to the
agent's observability and produces slightly smaller
`gap_to_oracle` values for those cells.  This is correct — an
oracle the agent could never match isn't a useful baseline — but
readers comparing pre-v19 and post-v19 summary.json files for
paper.yaml / smoke_v8 / coverage_stress should know about the
gap-shift in C3/C4.  The C1/C2 oracle is unchanged.

If a future paper draft prefers the previous "Z-cheating oracle"
semantics for C3/C4 (treating it as an unattainable upper bound),
the marginalisation can be gated on `horizon == 1` or behind an
`oracle_can_see_z` flag.  v19 applies it uniformly.

## The deliberate `pi_b_known` collapse

UCB / UCB± / UCB+ / RCT are on-policy: they collect their own
rollouts via `algo.select_action`, not the behaviour-policy's
actions.  The runner's `pi_b_known` cell-axis only gates the IPW
correction in the *off-policy* training path, which on-policy
algorithms never reach.  Consequently, in `paper_bandit`:

* C1 ≡ C2 (both `expose_z=True`; `pi_b_known` axis inert)
* C3 ≡ C4 (both `expose_z=False`; same)

The four-cell layout is retained in the YAML for matrix uniformity
with `paper.yaml`.  Bandit results validate the `expose_z` axis only;
the full four-cell taxonomy is exercised in the sequential env.

## Future work

If the paper needs a four-cell bandit demonstration, an off-policy
bandit algorithm (e.g. an offline IPW estimator) would have to be
added — that is not in v19.  The figure-rendering bugs (regret
curves, headline panel) are separate fixes deferred to v20.
