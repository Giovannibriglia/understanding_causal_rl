"""v17: ``_evaluate_gap_metrics`` is called once per *unique* tick in
``train_ticks ∪ eval_ticks``, not once per (tick, branch) pair.

Pre-v17, when ``n_checkpoints_train == n_checkpoints_eval`` (the
default for most configs), the train and eval tick sets coincide and
the runner called ``_evaluate_gap_metrics`` twice per shared tick —
once with seed offset ``+ 40_000`` (train branch), once with ``+ 50_000``
(eval branch).  The two values were i.i.d. estimates of the same
quantity; nothing downstream relied on having two distinct seeds.

v17 caches the train-branch result in ``self._last_gap_metrics`` and
reuses it for the eval-branch log call when the same tick is also
an eval tick.  The dedup applies only when ticks overlap; on
disjoint tick sets the eval branch still computes its own gap
metrics at seed offset ``+ 50_000``.

See ``docs/v17_gap_metric_dedup.md``.
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
    very small budget so the test runs in seconds."""
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


def test_gap_metrics_deduplicated_on_overlapping_ticks(tmp_path: Path) -> None:
    """When ``n_checkpoints_train == n_checkpoints_eval`` and tick sets
    coincide, ``_evaluate_gap_metrics`` is called once per shared tick.

    With ``total_frames=40`` and ``n_train = n_eval = 3``,
    ``checkpoint_ticks(40, 3) = {1, 20, 40}`` for both.  Expected call
    count: 3 (once per unique tick), not 6 (3 train + 3 eval).
    """
    cfg = _bandit_cfg(tmp_path / "overlap", n_train=3, n_eval=3)
    train_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_train)
    eval_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_eval)
    union = train_ticks | eval_ticks
    intersection = train_ticks & eval_ticks
    assert len(intersection) == len(union), (
        "fixture broken: tick sets should fully overlap for this test"
    )
    expected_calls = len(union)
    pre_v17_calls = len(train_ticks) + len(eval_ticks)
    assert pre_v17_calls > expected_calls, (
        "fixture broken: dedup should reduce call count"
    )

    runner = BenchmarkRunner(cfg)
    with patch.object(
        runner,
        "_evaluate_gap_metrics",
        wraps=runner._evaluate_gap_metrics,
    ) as wrapped:
        runner.run()
    assert wrapped.call_count == expected_calls, (
        f"expected {expected_calls} gap-metric calls (one per unique tick "
        f"in {sorted(union)}); got {wrapped.call_count} — pre-v17 dedup "
        f"would have shown {pre_v17_calls}"
    )


def test_gap_metrics_called_independently_when_ticks_disjoint(
    tmp_path: Path,
) -> None:
    """When tick sets only partially overlap, the eval branch computes
    its own gap metrics for ticks not shared with the train set.

    With ``total_frames=40``, ``n_train=4``, ``n_eval=2``:
    * ``checkpoint_ticks(40, 4) = {1, 13, 27, 40}``
    * ``checkpoint_ticks(40, 2) = {1, 40}``
    Union has 4 elements (train ∪ eval).  The eval branch dedups at
    ticks 1 and 40 (shared) and computes fresh at neither in this
    case (eval has no exclusive ticks).  Train fires fresh at all 4
    of its ticks.  Total: 4 calls.
    """
    cfg = _bandit_cfg(tmp_path / "disjoint", n_train=4, n_eval=2)
    train_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_train)
    eval_ticks = checkpoint_ticks(cfg.total_frames, cfg.n_checkpoints_eval)
    union = train_ticks | eval_ticks
    eval_only = eval_ticks - train_ticks
    expected_calls = len(train_ticks) + len(eval_only)
    assert expected_calls == len(union), (
        "v17 contract: total calls equals tick-set-union cardinality"
    )

    runner = BenchmarkRunner(cfg)
    with patch.object(
        runner,
        "_evaluate_gap_metrics",
        wraps=runner._evaluate_gap_metrics,
    ) as wrapped:
        runner.run()
    assert wrapped.call_count == expected_calls, (
        f"expected {expected_calls} gap-metric calls "
        f"(|train| + |eval - train| = {len(train_ticks)} + {len(eval_only)} "
        f"= {expected_calls}); got {wrapped.call_count}"
    )
