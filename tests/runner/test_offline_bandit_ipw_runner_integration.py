"""v21 Phase 3: the off-policy training loop populates
``behaviour_logprob`` in the batch dict.

The runner builds the per-step batch differently depending on
``pi_b_known``.  When True it pulls true logprobs straight from the
buffer (``batch_obj.behaviour_logprob``); when False it falls back to
the fitted propensity model (``self._propensity_model``).  Either way
the batch dict carries a non-None tensor — algos like
``OfflineBanditIPW`` consume it; existing algos ignore it.

This test guards Phase 3's runner-side plumbing.  It runs the bandit
``offline_bandit_ipw`` algo end-to-end on a tiny smoke fixture in
both ``pi_b_known=True`` (cell=1) and ``pi_b_known=False`` (cell=2)
cells and asserts:
* The run completes without exception.
* ``eval.csv``'s final ``eval_return_mean`` is finite.
* The algo's ``update`` saw a non-None ``behaviour_logprob`` at least
  once.  Spy via ``patch.object(... wraps=...)``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from causal_rl.algos.off_policy.offline_bandit_ipw import OfflineBanditIPW
from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _smoke_cfg(out: Path, *, cell: int) -> RunnerConfig:
    return RunnerConfig(
        cell=cell,
        env_name="bandit-tabular-sepsis-v0",
        algorithm="offline_bandit_ipw",
        behaviour="reward_aligned",
        seed=0,
        total_frames=200,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=100,
        offline_updates=10,
        alpha_conf=2.0,
        bias_strength=1.0,
        horizon=1,
        rollout_horizon=1,
        eval_perturbations=False,
        perturbation_grid_size=0,
        n_bootstrap=2,
        n_samples_gap=8,
        n_eval_episodes=1,
    )


def _last_eval_return(run_dir: Path) -> float:
    with (run_dir / "eval.csv").open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["eval_return_mean"])


@pytest.mark.parametrize("cell", [1, 2])
def test_runner_populates_behaviour_logprob_for_offline_bandit_ipw(
    tmp_path: Path, cell: int
) -> None:
    cfg = _smoke_cfg(tmp_path / f"cell{cell}", cell=cell)

    saw_logprob: list[bool] = []
    real_update = OfflineBanditIPW.update

    def spy(self: OfflineBanditIPW, batch: dict[str, object]) -> dict[str, float]:
        saw_logprob.append(batch.get("behaviour_logprob") is not None)
        return real_update(self, batch)

    with patch.object(OfflineBanditIPW, "update", spy):
        runner = BenchmarkRunner(cfg)
        runner.run()

    assert saw_logprob, "fixture broken: OfflineBanditIPW.update was never called"
    assert any(saw_logprob), (
        f"runner never populated behaviour_logprob in the batch dict for "
        f"cell={cell} (pi_b_known={cfg.cell == 1 or cfg.cell == 3}); "
        f"saw {sum(saw_logprob)}/{len(saw_logprob)} non-None"
    )

    eval_return = _last_eval_return(cfg.output_dir)
    assert math.isfinite(eval_return), (
        f"eval_return_mean is not finite ({eval_return}) on cell={cell} — "
        f"the offline bandit IPW path produced NaN/inf rewards"
    )
