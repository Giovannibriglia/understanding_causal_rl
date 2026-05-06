"""v18: ``_compute_bound_metrics`` is called only at checkpoint ticks
(``train_ticks ∪ eval_ticks``), not on every per-step update.

Pre-v18 the on-policy training loop in ``_run_online`` called
``_compute_bound_metrics`` once per ``should_update`` iteration.  With
``rollout_horizon=1`` (the bandit default) every tick is an update
tick, so the bound diagnostic ran ``total_frames`` times — 64,000 calls
per ``paper_bandit`` run.  ``_bound_metrics`` is read only by
``_log_train`` and ``_log_eval``, both of which are themselves
checkpoint-gated, so all but ``|train ∪ eval|`` of those calls were
overwritten before any consumer read them.

v18 gates the call site at ``runner.py:1041`` with
``if tick in train_ticks or tick in eval_ticks``.  Output CSVs are
unchanged (the gate fires whenever a log branch fires); the only
observable change is call count.

The off-policy site at ``runner.py:1141`` is not gated — it lives
outside the per-step loop in ``_run_offline`` (called once per run at
buffer-collection time) so it was never on the hot path.

See ``docs/v18_bound_metric_gating.md``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig
from causal_rl.runner.scheduling import checkpoint_ticks


def _bandit_cfg(
    out: Path, *, n_train: int, n_eval: int, total_frames: int = 40
) -> RunnerConfig:
    """Cheapest possible discrete-action config: bandit env, horizon 1,
    very small budget so the test runs in seconds.  Mirrors the
    ``_bandit_cfg`` helper in ``test_gap_metric_dedup.py``."""
    return RunnerConfig(
        cell=1,
        env_name="bandit-tabular-sepsis-v0",
        algorithm="ucb",
        behaviour="uniform",
        seed=0,
        total_frames=total_frames,
        n_checkpoints_train=n_train,
        n_checkpoints_eval=n_eval,
        output_dir=out,
        device="cpu",
        n_envs=4,
        batch_size=4,
        offline_transitions=20,
        offline_updates=4,
        eval_perturbations=False,
        perturbation_grid_size=0,
        horizon=1,
        rollout_horizon=1,
        n_bootstrap=2,
        n_samples_gap=8,
        n_eval_episodes=1,
    )


def test_bound_metrics_called_only_at_checkpoints(tmp_path: Path) -> None:
    """When tick sets fully overlap, ``_compute_bound_metrics`` runs
    once per shared tick.

    With ``total_frames=40`` and ``n_train = n_eval = 3``,
    ``checkpoint_ticks(40, 3) = {1, 20, 40}``.  Expected on-policy call
    count: 3 (one per tick in the union).  Pre-v18 the on-policy loop
    fired the diagnostic on every update — i.e. 40 times — so a
    successful gate cuts call count by ~13×.
    """
    cfg = _bandit_cfg(tmp_path / "overlap", n_train=3, n_eval=3)
    train_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_train)
    eval_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_eval)
    union = train_ticks | eval_ticks
    expected_online_calls = len(union)
    pre_v18_online_calls = cfg.total_frames
    assert pre_v18_online_calls > expected_online_calls, (
        "fixture broken: gate should reduce call count"
    )

    runner = BenchmarkRunner(cfg)
    with patch.object(
        runner,
        "_compute_bound_metrics",
        wraps=runner._compute_bound_metrics,
    ) as wrapped:
        runner.run()
    # The off-policy ``_run_offline`` path is not exercised by an
    # on-policy ``ucb`` run, so every call counted here comes from the
    # gated on-policy site.
    assert wrapped.call_count == expected_online_calls, (
        f"expected {expected_online_calls} bound-metric calls (one per "
        f"unique tick in {sorted(union)}); got {wrapped.call_count} — "
        f"pre-v18 the on-policy loop would have fired "
        f"{pre_v18_online_calls} times"
    )


def test_bound_metrics_computed_when_train_or_eval_tick_fires(
    tmp_path: Path,
) -> None:
    """When tick sets only partially overlap, the gate fires whenever
    *either* a train tick or an eval tick lands on the current step.

    With ``total_frames=40``, ``n_train=3``, ``n_eval=4``:
    * ``checkpoint_ticks(40, 3) = {1, 20, 40}``
    * ``checkpoint_ticks(40, 4) = {1, 13, 27, 40}``
    Their union has 5 elements (strictly more than either tick set
    alone) — the gate must consult *both* sets for its predicate to
    fire correctly here.  Pre-v18 the on-policy loop would have fired
    40 times regardless.
    """
    cfg = _bandit_cfg(tmp_path / "disjoint", n_train=3, n_eval=4)
    train_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_train)
    eval_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_eval)
    union = train_ticks | eval_ticks
    expected_online_calls = len(union)
    assert expected_online_calls > max(len(train_ticks), len(eval_ticks)), (
        "fixture broken: union should be strictly larger than either "
        "tick set — otherwise the test is indistinguishable from a "
        "single-set gate"
    )

    runner = BenchmarkRunner(cfg)
    with patch.object(
        runner,
        "_compute_bound_metrics",
        wraps=runner._compute_bound_metrics,
    ) as wrapped:
        runner.run()
    assert wrapped.call_count == expected_online_calls, (
        f"expected {expected_online_calls} bound-metric calls "
        f"(|train ∪ eval| over {sorted(union)}); got "
        f"{wrapped.call_count}"
    )
