# v25 — Simpson's-paradox bandit reward profile

## 1. Why v25

The `paper_bandit_offpolicy_20260508_120225` run (git_sha `d82a1f3`,
the v22 empirical step) produced pooled CV r² = 0.635 — supporting
claim 4 — but the per-cell-per-algorithm summary showed
`offline_bandit_naive` and `offline_bandit_ipw` produce *bit-identical*
aggregate stats in C1, C3, C4 (and a tiny ~0.025 gap difference in
C2) across 40 seeds.  Two algorithms with different update rules
producing the same eval returns is not "IPW correction is small" —
it's "naive and IPW recover the same argmax on this env."

Diagnosis (verified end-to-end before v25):

* The IPW algo is correctly implemented (v21 unit tests confirm
  argmax recovery on a synthetic confounded batch).
* The runner plumbing is correct — `expose_pi_b=cell.pi_b_known`
  flows true behaviour log-probs to IPW in C1/C3, and propensity-
  model estimates in C2/C4 with mean recovery KL ≈ 0.0024.
* The `TabularSepsisEnv` reward structure is the limiting factor.
  The smooth sigmoid + clipping in `_load_or_build_reward` doesn't
  admit argmax-flipping confounding — naive and IPW recover the same
  arm because the per-arm reward distributions are similar enough
  that the marginalised optimum lies adjacent to each per-Z optimum.

For the paper's bandit section to demonstrate a controlled-pair
failure case (same buffer, IPW recovers, naive fails), the env needs
a Simpson's-paradox structure.

## 2. The Simpson's-paradox design

`_build_reward_simpson` in `tabular_sepsis.py` constructs:

| arm     | Z=0    | Z=1    | marginal |
|---------|--------|--------|----------|
| 0       | P_HIGH | P_LOW  | (H+L)/2  |
| 1..N-2  | P_MID  | P_MID  | P_MID    |
| N-1     | P_LOW  | P_HIGH | (H+L)/2  |

with `P_HIGH=0.85`, `P_LOW=0.10`, `P_MID=0.55`.

* Marginal under uniform P(Z): side arms = 0.475, middle arms = 0.55.
* Middle arms strictly dominate the side arms by 0.075 under
  marginal P(Z).
* A behaviour policy with `expose_latent=True` and `α_conf > 0`
  picks the per-Z preferred side arm: arm 0 when Z=0, arm N-1 when
  Z=1.
* Naive sample-mean of arm 0 is biased upward — only sees arm 0 when
  Z=0 (where it has reward P_HIGH).  Naive's argmax: arm 0 or arm N-1.
* IPW reweights each transition by `1/π_b(a|Z)`, recovers the
  marginal: arms 0 and N-1 estimate ≈ 0.475 = -0.05 on [-1, +1];
  middle arms estimate ≈ 0.55 = +0.10.  IPW's argmax: a middle arm.

## 3. The unit-test gate

`tests/algos/test_simpson_argmax_flip.py` is a pre-compute gate.
Three tests:

* `test_simpson_env_flips_naive_vs_ipw_argmax` — the headline
  contract.  Builds a confounded batch from the simpson env with
  Z-peeking behaviour at `p_preferred=0.3`, asserts naive picks a
  side arm and IPW picks a middle arm.
* `test_simpson_naive_picks_side_arm_with_biased_estimated_mean` —
  defence-in-depth.  Naive's sample-mean of the picked side arm
  matches the analytical conditional bias E[r | a=side] = 0.325
  within Bernoulli noise.
* `test_smooth_profile_does_not_flip_argmax` — pins the pre-v25
  behaviour (smooth profile produces no argmax flip on the same
  fixture) so a future smooth-profile change that coincidentally
  creates a flip would be flagged.

### Iteration log

The brief's first-attempt fixture used `p_preferred=0.9` (strong
Z-peeking).  This produced inverse weights ~70 (`1 / (0.1 / 7)`) on
the rare-Z arm samples, which the algo's default `ipw_clip=10`
suppressed — IPW's arm-0 mu_hat collapsed to 0.51 instead of the
analytical -0.05, causing the gate to fail.

Fix: lower `p_preferred=0.3` so the rare-side weight `1 / (0.7 / 7)`
sits exactly at 10.  IPW is now unbiased on the fixture without
overriding the default clip.

### Final tuned parameters

```
P_HIGH = 0.85   # side arm under preferred Z
P_LOW  = 0.10   # side arm under non-preferred Z
P_MID  = 0.55   # middle arms under both Z
p_preferred = 0.3   # Z-peeking behaviour preference (in fixture only)
```

Behaviour: with `p_preferred=0.3` and uniform P(Z),
`P(z=0 | a=0) = 0.75`.  Naive bias on arm 0:
`E[r | a=0] = 0.75 * 0.7 + 0.25 * -0.8 = 0.325` on [-1, +1].
IPW recovers the marginal: `arm 0 ≈ -0.05`, middle arms ≈ +0.10.
The flip holds despite the small (0.075) marginal gap because IPW
is unbiased on this fixture.

## 4. Backward-compat invariant

The new reward profile is opt-in.  `confounding_profile="smooth"`
is the default everywhere:

* `RunnerConfig.confounding_profile = "smooth"`
* `MatrixConfig.confounding_profile = "smooth"`
* `RunConfigModel.confounding_profile = "smooth"`
* `TabularSepsisEnv.__init__(..., confounding_profile="smooth")`

`scripts/run_full_matrix._build_cmd` appends `--confounding-profile`
to the per-run command only when non-default — so every existing
config's `run_single.py` invocation is byte-identical to pre-v25.

`tests/envs/test_tabular_sepsis_simpson.py::test_smooth_profile_byte_identical`
is the regression guard: it asserts that explicit
`confounding_profile="smooth"` produces a reward / transition tensor
bit-identical to the default and that the result is cell-independent.

## 5. Expected per-cell signature on the rerun

Once v25 is merged and `paper_bandit_offpolicy.yaml` re-runs with
`confounding_profile: simpson`:

* **C1 (`expose_z=T, pi_b_known=T`)**: naive's `gap_to_oracle` should
  be large (≥ 0.3 on the [-1, +1] reward scale) under
  `reward_aligned × α_conf=4`.  IPW with true logprobs should be
  small (~ 0.1).  Headline contrast.
* **C3 (`expose_z=F, pi_b_known=T`)**: similar contrast, weaker
  because the marginalised oracle is itself weaker (v19 fix).
* **C2, C4 (`pi_b_known=F`)**: IPW's contrast is dampened by the
  propensity-model estimation error; naive's gap is unchanged
  relative to C1/C3.
* **Pooled R²**: should increase over the smooth-profile run (0.635)
  since the algorithm-axis variance is now non-trivial — IPW vs
  naive contributes within-cell predictive power that Δ_TV can pick
  up.

## 6. Scope of the claim

This run does *not* claim the framework "always works."  It claims
the diagnostic taxonomy + IPW correction together recover the
do-optimum on a hand-crafted confounded bandit where naive
demonstrably fails.  Controlled-pair existence proof, not generality
claim.  The smooth-profile result remains in the paper as the
"low-confounding" baseline; the simpson-profile result is the
"high-confounding" failure case.
