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
- `delta_tv_train_policy` (stochastic train policy) for occupancy-shift dynamics.

Expected shapes:
- `id`: decays toward ~0,
- `partial_id`: decays then plateaus,
- `non_id`: plateaus high.

Pre-committed check: in partial-id regimes, plateau height should track `bound_width_mean / 2`.
