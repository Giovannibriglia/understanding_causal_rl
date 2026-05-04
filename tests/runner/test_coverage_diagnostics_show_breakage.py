"""Phase 5.2 acceptance: histogram-based coverage flips the sign of the
masked-vs-unmasked comparison.

This is the test that would have caught the PR 17 bug.

Background.  PR 17 added ``forbidden_actions`` to the offline collector,
intending to produce coverage-broken runs.  The mechanism does block
actions correctly, but the per-row ``log_prob`` values written to the
buffer are uniform over the *allowed* subset — so ``min_propensity`` and
``ess_ratio`` read off a perfectly uniform distribution and report the
masked buffer as *more healthy* than an unmasked baseline.

The v12 fix adds ``coverage_action_freq_min`` and
``coverage_ess_histogram`` columns that read the empirical action
histogram directly.  This test verifies they invert the comparison —
the masked run has *worse* histogram-based coverage even though the
log-prob diagnostics still report it as healthier.
"""

from __future__ import annotations

import csv
from pathlib import Path

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _smoke_cfg(out: Path, forbidden: list[int]) -> RunnerConfig:
    return RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="dqn",
        behaviour="reward_aligned",
        seed=0,
        total_frames=200,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=400,
        offline_updates=40,
        alpha_conf=2.0,
        bias_strength=1.0,
        forbidden_actions=tuple(forbidden),
        eval_perturbations=False,
        perturbation_grid_size=0,
    )


def _read_last_eval_row(run_dir: Path) -> dict[str, str]:
    with (run_dir / "eval.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def test_masked_run_has_lower_histogram_coverage(tmp_path: Path) -> None:
    """Histogram-based coverage flags the masked run; log-prob-based
    coverage continues to mis-report it as healthy.

    The brief specifies the contract explicitly:
    * masked runs should have ``coverage_ess_histogram < 0.55``;
    * unmasked runs should have ``coverage_ess_histogram > 0.55``;
    * masked ``coverage_action_freq_min`` should be < 0.001 (forbidden
      arms have count 0);
    * unmasked ``coverage_action_freq_min`` should be > 0.05 (every arm
      gets non-trivial mass under reward_aligned at bs=1).
    """
    BenchmarkRunner(_smoke_cfg(tmp_path / "unmasked", forbidden=[])).run()
    BenchmarkRunner(
        _smoke_cfg(tmp_path / "masked", forbidden=[0, 1, 2, 3])
    ).run()

    unmasked = _read_last_eval_row(tmp_path / "unmasked")
    masked = _read_last_eval_row(tmp_path / "masked")

    masked_hist = float(masked["coverage_ess_histogram"])
    unmasked_hist = float(unmasked["coverage_ess_histogram"])
    masked_freq = float(masked["coverage_action_freq_min"])
    unmasked_freq = float(unmasked["coverage_action_freq_min"])

    # Sanity: log-prob ess_ratio is the *broken* signal.  At minimum it
    # should not flag the masked run as worse — that's the whole point
    # of the v12 bug.  We assert the broken behaviour is still present
    # so this test fails loudly if someone "fixes" the wrong place.
    masked_ess_logprob = float(masked["ess_ratio"])
    unmasked_ess_logprob = float(unmasked["ess_ratio"])
    assert masked_ess_logprob >= unmasked_ess_logprob - 0.01, (
        "log-prob ess_ratio is supposed to under-report coverage breakage "
        "(see PR 18 brief); if this assertion fires the bug semantics changed."
    )

    # The actual fix: histogram-based coverage shows masked is worse.
    assert masked_hist < 0.55, (
        f"masked coverage_ess_histogram should be < 0.55, got {masked_hist:.3f}"
    )
    assert unmasked_hist > 0.55, (
        f"unmasked coverage_ess_histogram should be > 0.55, got {unmasked_hist:.3f}"
    )
    assert masked_freq < 0.001, (
        f"masked coverage_action_freq_min should be < 0.001 (forbidden "
        f"arms have count 0), got {masked_freq:.4f}"
    )
    assert unmasked_freq > 0.05, (
        f"unmasked coverage_action_freq_min should be > 0.05, got {unmasked_freq:.4f}"
    )
