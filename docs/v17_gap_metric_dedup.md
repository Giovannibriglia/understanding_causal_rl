# v17 — gap-metric dedup on overlapping ticks

When ``n_checkpoints_train == n_checkpoints_eval`` (the default for
most configs, including ``paper_bandit``), the train and eval tick
sets produced by ``checkpoint_ticks`` are identical.  Pre-v17 each
overlapping tick called ``_evaluate_gap_metrics`` twice — once at
the train-log call site (seed offset ``+ 40_000``) and once at the
eval-log call site (seed offset ``+ 50_000``).  The two were
statistically i.i.d. estimates of the same quantity.

v17 caches the train-branch result in ``self._last_gap_metrics`` and
reuses it when the same tick is also an eval tick.  The new code:

```python
if tick in eval_ticks:
    eval_returns = self._evaluate_policy(...)
    oracle_returns = self._evaluate_oracle(...)
    if tick in train_ticks:
        gap_metrics = self._last_gap_metrics
    else:
        gap_metrics = self._evaluate_gap_metrics(seed=... + 50_000)
    self._log_eval(...)
```

## Backward-compatibility note

On configs where train and eval ticks coincide (most of them — see
``checkpoint_ticks`` for the default behaviour), v17 ``eval.csv`` rows
report the same ``delta_tv`` / ``delta_kl`` / etc. values as the
corresponding ``train.csv`` rows at the same ``step``.  Pre-v17 the
two would have differed by Monte-Carlo noise (different seed offsets
draw different sample tensors).

No downstream consumer relies on the per-tick i.i.d.-pair structure:

* ``make_summary.py`` reads only the *last* row of ``eval.csv`` per run.
* The headline regression treats each eval row as a single sample.
* The figure pipeline plots train and eval as separate series (with
  the dedup, the eval series's value at each overlapping tick equals
  the train value — but visually this just removes a small amount of
  noise from the plot).

When ``n_checkpoints_train != n_checkpoints_eval`` (or the two tick
sets diverge for any reason), the eval branch still computes its own
gap metrics at seed offset ``+ 50_000`` — the dedup applies only when
the tick is shared.
