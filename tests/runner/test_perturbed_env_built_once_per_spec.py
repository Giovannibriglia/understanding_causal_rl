"""Phase 2 acceptance: the perturbed env for each spec is built once, not three
times.

Before v10 the runner constructed a perturbed env three separate times per
spec — one for the rollout, one for D_env_KS, one for D_bisim.  Each
construction allocated a 1440×8×1440 transition tensor (≈ 66 MB, ~400 ms).
"""

from __future__ import annotations

from pathlib import Path

from causal_rl.envs.perturbations import TABULAR_DIAGONAL
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv
from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _smoke_cfg(out: Path, grid_size: int) -> RunnerConfig:
    return RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="cql",
        behaviour="uniform",
        seed=0,
        total_frames=40,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=80,
        offline_updates=40,
        n_eval_episodes=1,
        perturbation_grid_size=grid_size,
        eval_perturbations=True,
    )


def test_perturbed_env_built_once_per_spec(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    grid_size = 3
    n_specs = min(grid_size, len(TABULAR_DIAGONAL))

    counter = {"n": 0}
    original_init = TabularSepsisEnv.__init__

    def counting_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(TabularSepsisEnv, "__init__", counting_init)

    runner = BenchmarkRunner(_smoke_cfg(tmp_path / "spec3", grid_size))
    counter_before = counter["n"]
    runner._evaluate_perturbations(
        step=runner.config.total_frames,
        episode=0,
        seed=42,
    )
    constructions_during = counter["n"] - counter_before

    # 1 base env + n_specs perturbed envs = n_specs + 1 constructions.
    # The previous implementation produced 3 * n_specs + 1.
    assert constructions_during <= n_specs + 1, (
        f"perturbation eval constructed {constructions_during} envs; "
        f"expected at most {n_specs + 1} (1 base + 1 per spec)."
    )
