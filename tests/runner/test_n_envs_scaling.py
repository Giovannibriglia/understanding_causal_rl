"""Tests for n_envs parallelisation correctness and scaling."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig


def _make_config(tmp_path: Path, n_envs: int, eval_n_envs: int | None = None) -> RunnerConfig:
    return RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="cql",
        behaviour="uniform",
        seed=42,
        total_frames=40,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=tmp_path / f"run_{n_envs}",
        device="cpu",
        n_envs=n_envs,
        eval_n_envs=eval_n_envs,
        batch_size=n_envs,
        offline_transitions=n_envs * 2,
        offline_updates=10,
        n_bootstrap=0,
        n_samples_gap=20,
    )


def test_meta_total_transitions(tmp_path: Path) -> None:
    """meta.json logs total_transitions = total_frames * n_envs."""
    n_envs = 8
    total_frames = 40
    cfg = _make_config(tmp_path, n_envs=n_envs)
    runner = BenchmarkRunner(cfg)
    runner.run()
    meta = json.loads((tmp_path / "run_8" / "meta.json").read_text())
    assert meta["total_transitions"] == total_frames * n_envs
    assert meta["n_envs"] == n_envs
    assert meta["total_frames"] == total_frames


def test_eval_n_envs_separate(tmp_path: Path) -> None:
    """eval_n_envs is used for evaluation envs independently of n_envs."""
    cfg = _make_config(tmp_path, n_envs=8, eval_n_envs=4)
    assert cfg.eval_n_envs == 4
    runner = BenchmarkRunner(cfg)
    assert runner._eval_n_envs == 4
    runner.run()
    # Verify run completed successfully
    assert (tmp_path / "run_8" / "eval.csv").exists()


def test_eval_n_envs_defaults_to_n_envs(tmp_path: Path) -> None:
    """When eval_n_envs is None, _eval_n_envs falls back to n_envs."""
    cfg = _make_config(tmp_path, n_envs=8, eval_n_envs=None)
    runner = BenchmarkRunner(cfg)
    assert runner._eval_n_envs == 8


def test_larger_n_envs_completes(tmp_path: Path) -> None:
    """n_envs=32 run completes in under 30s on CPU (parallelism smoke check)."""
    cfg = RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="cql",
        behaviour="uniform",
        seed=0,
        total_frames=40,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=tmp_path / "big_run",
        device="cpu",
        n_envs=32,
        batch_size=32,
        offline_transitions=128,
        offline_updates=10,
        n_bootstrap=0,
        n_samples_gap=20,
    )
    t0 = time.time()
    runner = BenchmarkRunner(cfg)
    runner.run()
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"Run took {elapsed:.1f}s — expected < 30s"


def test_n_envs_sanity_check_warns(tmp_path: Path) -> None:
    """A very large n_envs * n_bootstrap * n_samples triggers a warning."""
    cfg = RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="cql",
        behaviour="uniform",
        seed=0,
        total_frames=40,
        n_checkpoints_train=1,
        n_checkpoints_eval=1,
        output_dir=tmp_path / "warn_run",
        device="cpu",
        n_envs=16,
        batch_size=16,
        offline_transitions=64,
        offline_updates=5,
        # n_envs * n_bootstrap * n_samples = 16 * 200000 * 200 >> 5e8
        n_bootstrap=200000,
        n_samples_gap=200,
    )
    with pytest.warns(UserWarning, match="Clamping n_bootstrap"):
        runner = BenchmarkRunner(cfg)
    assert runner._effective_n_bootstrap < cfg.n_bootstrap
