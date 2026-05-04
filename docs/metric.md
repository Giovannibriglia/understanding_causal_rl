# Metric

$$
\Delta_\varphi(\pi)=\mathbb{E}_{(s,a)\sim d^\pi}\left[\varphi\!\left(\hat P(R\mid a,s),\hat P(R\mid do(a),s)\right)\right].
$$

The key point is policy dependence: as training changes occupancy \(d^\pi\), \(\widehat{\Delta}\) changes.

Three forces shape curves over training:
1. estimator noise (early high variance),
2. occupancy concentration (can transiently increase measured gap),
3. identifiability floor (non-zero plateau in partial/non-id cells).

Dual logging:
- `delta_tv` (deterministic eval policy) for cross-algorithm comparability,
- `delta_tv_beh` (behaviour policy under the same eval seeds) as a reference
  point for the occupancy-shift contribution.

Expected shapes:
- `id`: decays toward ~0,
- `partial_id`: decays then plateaus,
- `non_id`: plateaus high.

Pre-committed check: in partial-id regimes, plateau height should track `bound_width_mean / 2`.

## Metric inventory

| metric | failure mode | source |
| --- | --- | --- |
| `delta_tv` (and `delta_kl`/`delta_chi2`/`delta_sup`) | identifiability gap (operational) | this paper |
| `bound_width_mean`, per-arm `bound_lower_a*`, `bound_upper_a*`, `obs_arm_count_a*` | graphical ID — natural-bound width | Bareinboim §9.2 |
| `n_arms_with_informative_bound` | graphical ID — provably suboptimal arms | Bareinboim §9.2 |
| `min_propensity`, `ess_ratio`, `overlap_at_1e-2`, `tail_mass_top10pct` | recoverability / coverage | Hernán & Robins 2020 |
| `propensity_calibration_ece` | recoverability / model quality | Guo et al. 2017 |
| `pi_b_recovery_kl` (NEW) | recoverability / model quality | this paper |
| `backdoor_residual_mean`, `backdoor_residual_max` (NEW) | graphical ID — confirms backdoor sufficiency | Pearl 2009 |
| `target_support_overlap` (NEW) | recoverability — coverage of *deployment* support | Crump et al. 2009 |
| `cond_mi_r_z_given_sa` (NEW) | partial observability | Tishby & Zaslavsky 2015; Cheng et al. 2020 |
| `D_env_KS` | generalisation (reward shift) | this paper |
| `D_bisim` (NEW) | generalisation (reward + dynamics shift) | Castro & Precup 2010 |

Each of the four failure modes (graphical ID, statistical recoverability,
coverage, partial observability) now has a primary and a complementary
metric.  The headline-regression `partial_r2` column tells the reader
which axis contributed most to the observed return gap.

### Metric details

- `backdoor_residual` (`metrics/backdoor.py`): measures
  `E_naive[R | A=a] − Σ_z P̂(z) Ê[R | A=a, Z=z]`.  Goes to zero when the
  observed Z satisfies the back-door criterion; stays non-zero when Z is
  insufficient.  Returns NaN when Z is hidden by the cell.
- `target_support_overlap` (`metrics/coverage.py`): fraction of
  state-action pairs the trained policy would visit on which `π_b(a|s) > τ`.
  When this is < 0.5, off-policy estimation is unreliable regardless of
  dataset size.
- `pi_b_recovery_kl` (`metrics/coverage.py`): `KL(π_b_true ‖ π_b_predicted)`
  averaged over the buffer.  Sanity-checks that the propensity model fits
  the true behaviour; 0 when π_b is fully known, NaN when it cannot be
  computed.
- `cond_mi_r_z_given_sa` (`metrics/info_bottleneck.py`): CLUB-style upper
  bound (Cheng et al. 2020) on `I(R; Z | S, A)`.  In partial-id cells this
  number is the theoretical lower bound on the residual gap any agent
  can close without observing Z.
- `D_bisim` (`metrics/bisimulation.py`): Ferns–Castro–Precup bisimulation
  pseudometric, computed by 15-iteration fixed point on a 24×24 random
  state subset for runtime safety.  Bounds the policy-value gap
  `|V_π^M − V_π^M'| ≤ d / (1 − γ)`.
