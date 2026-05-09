# v26 — Reward-aligned policy fix (Z-peeking explorer)

## 1. What was wrong

The `paper_bandit_offpolicy_20260508_120225` matrix run (`d82a1f3`)
supported claim 4 numerically but produced bit-identical aggregate
stats for `offline_bandit_naive` and `offline_bandit_ipw` in cells
C1, C3, C4 across 40 seeds.  v25.1's gate test attempted to fix
this with a Simpson's-paradox env reward profile.  v25.2 audited
the runner's actual behaviour policy and found the simpson profile
couldn't help, because the bug was in the policy — not the env.

Three structural facts (`scripts/audit_alpha_conf_strength.py`,
`scripts/audit_alpha_conf_strength_sequential.py`,
`scripts/audit_temporal_z_leak_sequential.py`):

* `RewardAlignedExplorer.__init__` accepts `n_actions`,
  `requires_latent`, `beta=1.0`, `bias_strength` — **no `alpha_conf`
  argument**.  `α_conf` is an env-side parameter that modulates
  `observed_reward_prob` (used for divergence diagnostics) and
  `BanditView.collect_observational_data` (`ucb_plus` init only),
  but the rollout's actual reward flow is
  `do_reward(action) = reward_probs[_latent_state, a]` which depends
  on `_latent_state` (Z) alone.
* The collector calls `select_action(obs, latent=info["latent_U"])`.
  `latent_U` is a fresh `Bernoulli(0.5)` at every reset
  (`tabular_sepsis.py:266`), sampled **independently** of the
  reward-driving latent Z.
* No meaningful temporal Z-leak via transitions (v25.4): per-step
  Z-conditional ratios stay flat across a 20-step episode on both
  `expose_z=True` and `expose_z=False` cells.

Net: the policy peeks at U; U ⊥ Z; reward depends on Z; the policy
is reward-blind, and `α_conf=4` produces Z-conditional sample ratios
of 1.08 / 1.28:1 across bandit and sequential.  Buffers labelled
`partial_id`/`non_id` weren't actually Z-confounded.

## 2. What v26 changes

`src/causal_rl/behaviour/reward_aligned_z.py` — new
`RewardAlignedZExplorer` class:

* `latent_source = "latent_Z"` — the collector reads `info["latent_Z"]`
  rather than `info["latent_U"]` (default).
* `__init__` accepts `alpha_conf` and `env` as kwargs.  At
  construction time, computes per-Z preferred arms by marginalising
  `env.reward_probs[:, :, 1]` over the OBS_STATES rows of each Z slab
  and taking argmax.
* `select_action` builds per-env action weights `[1, …, exp(α), …, 1]`
  with the boosted index at the per-Z preferred arm; samples via
  `Categorical(probs)`.
* At `α_conf = 0` ⇒ all weights = 1 ⇒ uniform (verified by
  `test_alpha_zero_is_uniform`, counts within 3σ of 1000 across
  8000 samples).
* At `α_conf = 4` ⇒ preferred arm has `exp(4)/(exp(4) + 7) ≈ 88.6%`
  of the mass; Z-conditional sample ratio on a 4000-sample rollout
  is ~55–61:1 on the preferred arms (verified by
  `test_alpha_four_confounds`).

`src/causal_rl/behaviour/base.py` — adds `latent_source: str =
"latent_U"` class attribute.  Default matches all pre-v26 policies;
new explorer overrides.

`src/causal_rl/behaviour/registry.py` — registers `reward_aligned_z`
as a separate `BehaviourSpec` entry alongside the legacy
`reward_aligned`.  `depends_on_u=True` (gates the runner's
`expose_latent` flag, which the collector reads to decide whether
to pass any latent at all).

`src/causal_rl/runner/collector.py` —
`latent_key = getattr(behaviour_policy, "latent_source", "latent_U")`,
then `info.get(latent_key)` for both the per-step `select_action`
call (line 99) and the `latent_logged` write to the buffer (line
117).

`src/causal_rl/runner/runner.py` —
`_make_behaviour_with_bias` injects `alpha_conf` and `env` kwargs
specifically when the behaviour name is `reward_aligned_z`.  Other
explorers don't accept those kwargs, so the conditional is necessary;
no change to the legacy explorer's construction.

The on-policy gap-metric site at `runner.py:829` (the dual-policy
`delta_tv` measurement) also reads `info[behaviour_policy.latent_source]`
for symmetry.

## 3. What stays

`src/causal_rl/behaviour/reward_aligned.py` — the legacy
`RewardAlignedExplorer` is preserved.  It's now an honest
"U-peeking baseline" — useful for documenting what the old
taxonomy actually measured.

`configs/coverage_stress.yaml` — keeps the legacy `reward_aligned`.
Claim 6's coverage-break mechanism is action-masking (`forbidden_actions`)
and small-sample (`offline_transitions: 500`), both
`α_conf`-independent (v25.4 Phase 1).  The legacy near-uniform
behaviour adds noise that the regime sweep already accounts for;
no need to rerun.

## 4. Backward-compat invariants

* **`α_conf = 0` is uniform.**  Existing `α_conf = 0` summary rows
  shouldn't change in expectation when configs migrate from
  `reward_aligned` to `reward_aligned_z`.  Verified by
  `test_alpha_zero_is_uniform`.
* **`coverage_stress.yaml` summaries unchanged.**  The legacy
  explorer is unchanged; the v26 policy isn't used.
* **`α_conf > 0` rows for migrated configs WILL change.**  These
  are *intentional reruns* — the previous rows reflected
  near-uniform buffers.  Reruns produce the actually-confounded
  buffers the cell taxonomy describes.

## 5. Audit-trail link

The discovery sequence:

* `scripts/audit_alpha_conf_strength.py` (v25.2) — bandit;
  Z-ratio 1.08:1.
* `scripts/audit_alpha_conf_strength_sequential.py` (v25.3) —
  sequential at horizon=20; Z-ratio 1.28:1 across cells 2 and 4.
* `scripts/audit_temporal_z_leak_sequential.py` (v25.4) — temporal
  evolution; per-step ratios stay flat (no transition-mediated
  leak).
* `configs/coverage_stress.yaml` review (v25.4 Phase 1) — coverage-
  break mechanism is action-masking + small-sample; claim 6 intact.

These three scripts are preserved in this PR.  Their commit history
documents the audit trail.

## 6. v25 dead code (discarded)

The v25 prompt added a `confounding_profile` parameter to
`TabularSepsisEnv` with a `simpson` reward profile.  v25.1 retuned
its gate test fixture to verify the env profile worked in isolation.
v25.2 then discovered the bug was in the policy, not the env.

Discarded in v26:
* `TabularSepsisEnv.confounding_profile` parameter (and the dict-
  keyed `_REWARD_CACHE` that v25 introduced).
* `_build_reward_simpson` method.
* `tests/envs/test_tabular_sepsis_simpson.py` and
  `tests/algos/test_simpson_argmax_flip.py`.
* `--confounding-profile` and `--ipw-clip` CLI flags on
  `run_single.py` and the corresponding `MatrixConfig` /
  `RunConfigModel` fields.
* `docs/v25_simpson_bandit.md`.

The simpson profile would be a second confounding mechanism on top
of the now-fixed policy.  Adding it would conflate the two and
make per-cell results harder to interpret.  v23's reward cache
stays simple (`_REWARD_CACHE: Tensor | None`).

PR #31 (which introduced this dead code) is closed without merging.

## 7. v27 housekeeping note

v25.4 Phase 1 noticed that `coverage_stress.yaml`'s `biased` regime
is misnamed — it has identical knobs to `full` except
`offline_transitions` is 4000 instead of 8000.  No `bias_strength`
or `alpha_conf` differentiation.  Logged here for v27 as a
documentation/rename pass; not a v26 change.
