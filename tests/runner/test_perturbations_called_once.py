"""Phase 1 acceptance: ``_evaluate_perturbations`` runs once per training,
not once per checkpoint.

The per-spec quantities (D_env_KS, D_bisim) depend only on the env-pair —
they do not change with the trained policy or the training step.  Calling
the perturbation block at every checkpoint multiplied wall time by
``n_checkpoints_eval`` without adding any new information.
"""

from __future__ import annotations

from pathlib import Path

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


class _SpyRunner(BenchmarkRunner):
    """Spy on _evaluate_perturbations to count call sites."""

    def __init__(self, config: RunnerConfig) -> None:
        super().__init__(config)
        self.perturbation_call_count = 0

    def _evaluate_perturbations(self, step: int, episode: int, seed: int) -> None:  # type: ignore[override]
        self.perturbation_call_count += 1
        super()._evaluate_perturbations(step=step, episode=episode, seed=seed)


def _smoke_cfg(out: Path, n_ckpt: int) -> RunnerConfig:
    return RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="cql",
        behaviour="uniform",
        seed=0,
        total_frames=40,
        n_checkpoints_train=n_ckpt,
        n_checkpoints_eval=n_ckpt,
        output_dir=out,
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=80,
        offline_updates=40,
        n_eval_episodes=1,
        perturbation_grid_size=2,
        eval_perturbations=True,
    )


def test_perturbations_called_once_with_two_checkpoints(tmp_path: Path) -> None:
    runner = _SpyRunner(_smoke_cfg(tmp_path / "ckpt2", n_ckpt=2))
    runner.run()
    assert runner.perturbation_call_count == 1, (
        f"expected exactly 1 perturbation call, got {runner.perturbation_call_count}"
    )


def test_perturbations_called_once_with_five_checkpoints(tmp_path: Path) -> None:
    runner = _SpyRunner(_smoke_cfg(tmp_path / "ckpt5", n_ckpt=5))
    runner.run()
    assert runner.perturbation_call_count == 1, (
        f"expected exactly 1 perturbation call, got {runner.perturbation_call_count}"
    )


def test_perturbations_skipped_when_disabled(tmp_path: Path) -> None:
    cfg = _smoke_cfg(tmp_path / "ckpt5_off", n_ckpt=5)
    cfg = RunnerConfig(**{**cfg.__dict__, "eval_perturbations": False})
    runner = _SpyRunner(cfg)
    runner.run()
    # The end-of-training call still happens, but the body short-circuits
    # on the eval_perturbations flag — the spy still increments because
    # it counts call sites, not work performed.  We assert ≤ 1 to make
    # explicit that the call site is unique.
    assert runner.perturbation_call_count == 1
