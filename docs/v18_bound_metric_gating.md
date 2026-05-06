# v18 — bound-metric checkpoint gating + natural_bounds budget pipe-through

## Phase 1 — gate ``_compute_bound_metrics`` to checkpoints

The on-policy training loop in ``_run_online`` called
``_compute_bound_metrics`` once per ``should_update`` iteration.  With
``rollout_horizon=1`` (the bandit default) every tick is an update
tick, so the diagnostic ran ``total_frames`` times per run — 64,000
calls per ``paper_bandit`` cell.

``_bound_metrics`` is read only by ``_log_train`` and ``_log_eval``,
both of which are themselves checkpoint-gated.  Pre-v18 every call
that didn't land on a tick in ``train_ticks ∪ eval_ticks`` had its
output overwritten before any consumer read it.

v18 gates the call site at ``runner.py:1041``:

```python
if tick in train_ticks or tick in eval_ticks:
    self._compute_bound_metrics(action_batch, reward_batch)
```

Output CSVs are unchanged — the gate fires whenever a log branch
fires.  The only observable difference is call count.

The off-policy site at ``runner.py:1141`` lives outside the per-step
loop in ``_run_offline`` (called once per run at buffer-collection
time) and was never on the hot path; it is intentionally not gated.

## Phase 2 — pipe ``n_bootstrap`` through ``natural_bounds``

The bound-metric site at ``runner.py:490`` was passing
``n_bootstrap=200`` as a hard-coded literal, silently overriding the
v17 ``RunnerConfig.n_bootstrap`` knob and the memory-safety clamp
computed at ``__init__`` (line 168).  v18 uses
``self._effective_n_bootstrap`` so YAML/CLI overrides propagate to
this site, matching the gap-metric wiring.

The ucb_plus init site at ``runner.py:143`` is intentionally out of
scope — it runs once before training starts, with its own bootstrap
cost regime.

Plumbing the config through exposed a latent edge case:
``natural_bounds`` calls ``torch.quantile`` on the bootstrap replicate
tensor, which errors on empty input.  v17's schema explicitly allows
``n_bootstrap=0`` (``Field(ge=0)``) — pre-v18 the hard-coded ``200``
literal masked this; v18 must handle it.  The
``_compute_bound_metrics`` short-circuit now treats
``_effective_n_bootstrap <= 0`` the same way as a non-discrete env:
emit the all-NaN payload and skip the bootstrap.  CSV column shape
is unchanged.

## Backward compatibility

* Configs without ``n_bootstrap`` set continue to use the
  ``RunnerConfig`` default (200) — identical to the legacy literal.
* CSV columns and per-tick values are unchanged on every config where
  the diagnostic ran on a checkpoint tick anyway (i.e. all of them —
  the gate is a no-op for the values consumers actually read).
* Configs that do set ``n_bootstrap`` (e.g. ``paper_bandit``'s 50)
  now see that value reach this site too, slightly widening the
  bound CIs by Monte-Carlo noise.  The widening is invisible at paper
  precision (the bound diagnostic is averaged over arms × seeds).
