# Metric

The identifiability gap is
\[
\Delta_\varphi(\pi)=
\mathbb{E}_{(s,a)\sim d^\pi}\left[\varphi\left(
P(R\mid a,s),P(R\mid do(a),s)
\right)\right].
\]

We support:

- Total variation
- KL divergence
- Chi-squared divergence
- Supremum log-ratio

## Estimation protocol

- Checkpoint protocol: at each train/eval checkpoint, the runner rolls out the current policy deterministically and computes \(\widehat\Delta_\varphi\) on those visited \((s,a)\) pairs. This makes divergence curves policy-dependent (and therefore algorithm-dependent) within the same cell.
- Tabular env: plug-in estimate using `observed_reward_prob` vs oracle `do_reward` along the rollout occupancy. When behaviour propensities are known (offline archives), inverse-propensity weighting can be applied through logged behaviour log-probabilities.
- Continuous env: `do_reward` Monte-Carlo samples define an interventional reward distribution per state-action. Observed one-step rewards define observational reward distributions; TV, KL, \(\chi^2\), and sup-log-ratio are computed on Gaussianized discretizations, while marginal discrepancy is tracked via MMD.

## Limitations

- Finite-sample bias in plug-in estimators for low-support actions.
- Continuous divergences rely on Gaussianization/discretization of reward samples and can be sensitive to heavy tails.
- MMD bandwidth sensitivity in low-coverage settings.
- Unknown behaviour policy in cells 5/6/8 reduces correction quality.
