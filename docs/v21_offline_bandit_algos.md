# v21 — off-policy bandit algorithms (naive + IPW)

Two new bandit-only off-policy algorithms — `offline_bandit_naive` and
`offline_bandit_ipw` — and the runner-side plumbing they need to
consume per-batch behaviour log-probs.  v21 ships the infrastructure;
v22 will land `paper_bandit_offpolicy.yaml` and the empirical matrix
that uses these algos to populate the four-cell bandit taxonomy.

## Why

The post-v19/v20 bandit family uses on-policy UCB-family algorithms
(plus RCT) only.  UCB's `kind = "on_policy"`, so the runner's IPW
branch — which gates on `pi_b_known` — never fires.  The four-cell
taxonomy collapses to two cells in expectation (C1≡C2, C3≡C4),
documented in `docs/v19_bandit_cell_collapse.md`.

The two new algos are off-policy by construction.  They train on a
buffer of `(action, reward, behaviour_logprob)` tuples and differ
only in whether they apply IPW reweighting.  Together they exercise:

* `expose_z` — already exercised by the bandit env's reward
  marginalisation.
* `pi_b_known` — exercised by the *difference* between the two algos'
  treatment of `behaviour_logprob`, and by the runner's choice
  between true logprobs (when `pi_b_known=True`) and propensity-model
  estimates (when `pi_b_known=False`).

## Algorithms

* `OfflineBanditNaive` (`src/causal_rl/algos/off_policy/offline_bandit_naive.py`)
  — per-arm running mean of raw rewards, no propensity correction.
  Picks `argmax_a mu_hat[a]`.  Non-contextual: ignores `obs`
  deliberately.  Biased on confounded data — that's the point.
* `OfflineBanditIPW` (`src/causal_rl/algos/off_policy/offline_bandit_ipw.py`)
  — same per-arm structure with weights `w = exp(-log π_b(a))` clipped
  at `ipw_clip` (default 10, matching the runner's `compute_gap_metrics`
  IPW path).  Recovers `E[r | do(A=a)]` when behaviour log-probs are
  well-calibrated.  Falls back to naive when `behaviour_logprob` is
  absent from the batch.

Both are registered with `default_n_envs=8` (bandit-default, matching
UCB-family).

## Runner plumbing

`runner.py` line ~1290 now populates `batch["behaviour_logprob"]` in
the off-policy training loop:

```python
if self.cell_cfg.pi_b_known and batch_obj.behaviour_logprob is not None:
    behaviour_logprob = batch_obj.behaviour_logprob
elif self._propensity_model is not None:
    with torch.no_grad():
        full_logp = self._propensity_model(batch_obj.obs)
        behaviour_logprob = full_logp.gather(
            1, batch_obj.action.view(-1, 1).long()
        ).squeeze(-1)
```

The propensity model is fit unconditionally on the offline buffer
(line 1207), so the fallback is well-defined for any discrete-action
config.  Existing algos (DQN, CQL, ConfoundedDQN, ...) ignore the new
key.

## Cross-cell expected behaviour (preview for v22)

| Cell | expose_z | pi_b_known | naive | IPW (true π_b) | IPW (propensity model) |
|------|----------|------------|-------|----------------|-----------------------|
| 1    | True     | True       | biased on `α_conf>0` | recovers `E[r\|do(a)]` | unused (true π_b available) |
| 2    | True     | False      | biased on `α_conf>0` | unused | recovers up to model error |
| 3    | False    | True       | biased + Z marginalised | recovers marginal `E[r\|do(a)]` | unused |
| 4    | False    | False      | biased + Z marginalised | unused | recovers up to model error |

The "naive vs IPW" contrast is the controlled experiment v22 will
report.  The "true π_b vs propensity-model" contrast is the
`pi_b_known` axis the bandit family had been failing to exercise.

## Future work (v22+)

* `paper_bandit_offpolicy.yaml` — the bandit matrix that uses these
  algos.  Behaviour: `reward_aligned`, `α_conf` swept.  Algos: both
  new + UCB for reference.
* Contextual bandits (algos that condition on `obs`).  Out of scope
  for v21; the current algos are deliberately non-contextual.
* Replacing `compute_gap_metrics`'s IPW path.  That path uses its own
  weighting; the bandit IPW algo is a separate concern.
