from __future__ import annotations

import csv
from pathlib import Path

from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig
from causal_rl.runner.schemas import EVAL_COLUMNS, TRAIN_COLUMNS


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def test_runner_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = RunnerConfig(
        cell=3,
        env_name="tabular-sepsis-v0",
        algorithm="cql",
        behaviour="uniform",
        seed=0,
        total_frames=100,
        n_checkpoints_train=5,
        n_checkpoints_eval=5,
        output_dir=out,
        device="cpu",
        n_envs=16,
        batch_size=16,
        offline_transitions=200,
        offline_updates=50,
    )
    runner = BenchmarkRunner(cfg)
    runner.run()
    train_path = out / "train.csv"
    eval_path = out / "eval.csv"
    meta_path = out / "meta.json"
    assert train_path.exists()
    assert eval_path.exists()
    assert meta_path.exists()
    assert _read_header(train_path) == TRAIN_COLUMNS
    assert _read_header(eval_path) == EVAL_COLUMNS
