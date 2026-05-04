"""Phase 1.2 acceptance: offline coverage with pi_b_known=False.

The propensity model must be fit on the offline buffer before the coverage
metrics are recorded; otherwise ``min_propensity`` collapses to ``exp(0) = 1``
because the buffer's behaviour_logprob is zero-filled when π_b is hidden.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _last_train_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def test_offline_coverage_realistic_in_pi_b_unknown_cell(tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = RunnerConfig(
        cell=2,  # pi_b_known=False
        env_name="tabular-sepsis-v0",
        algorithm="dqn",
        behaviour="reward_aligned",
        seed=0,
        total_frames=40,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=500,
        offline_updates=40,
        alpha_conf=2.0,
        eval_perturbations=False,
    )
    BenchmarkRunner(cfg).run()
    last = _last_train_row(out / "train.csv")
    min_prop = float(last["min_propensity"])
    ess = float(last["ess_ratio"])
    ece_str = last.get("propensity_calibration_ece", "")
    ece = float(ece_str) if ece_str not in ("", "nan") else float("nan")

    assert min_prop < 0.9, (
        f"min_propensity={min_prop:.3f} indicates the propensity model isn't being used"
    )
    assert ess < 0.95, f"ess_ratio={ess:.3f} should not be near-degenerate"
    assert math.isfinite(ece), "propensity_calibration_ece must be finite when π_b is hidden"
    assert ece > 0.0, f"propensity_calibration_ece={ece:.3f} expected non-zero"

    # pi_b_recovery_kl should be finite (non-NaN) in pi_b_unknown cells: the
    # collector now records true behaviour log-probs in a side channel that
    # the runner uses for the KL even when expose_pi_b=False.
    kl_str = last.get("pi_b_recovery_kl", "")
    kl = float(kl_str) if kl_str not in ("", "nan") else float("nan")
    assert math.isfinite(kl), (
        f"pi_b_recovery_kl={kl_str!r} expected finite when pi_b_known=False"
    )
