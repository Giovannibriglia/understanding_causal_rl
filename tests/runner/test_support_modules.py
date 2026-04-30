from __future__ import annotations

import csv
from pathlib import Path

import torch

from causal_rl.config.schemas import MatrixConfig, RunConfigModel, load_yaml
from causal_rl.envs.registry import make
from causal_rl.replay.buffer import ReplayBuffer
from causal_rl.runner.collector import collect_offline_dataset
from causal_rl.runner.csv_logger import CSVLogger
from causal_rl.runner.runner import BenchmarkRunner, RunnerConfig
from causal_rl.utils.device import get_device
from causal_rl.utils.git import get_git_commit
from causal_rl.utils.seeding import set_seed


def test_config_models_and_yaml_loader(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "cells: [1]\nenvs: [tabular-sepsis-v0]\nalgorithms: [ppo]\nbehaviours: [uniform]\n"
    )
    matrix = MatrixConfig.model_validate(load_yaml(cfg_path))
    assert matrix.cells == [1]
    run = RunConfigModel.model_validate(
        {
            "cell": 1,
            "env": "tabular-sepsis-v0",
            "algorithm": "ppo",
            "behaviour": "uniform",
            "seed": 0,
            "output": "results/x",
        }
    )
    assert run.cell == 1


def test_replay_buffer_add_and_sample() -> None:
    buf = ReplayBuffer(
        capacity=32, obs_dim=6, act_dim=1, device=torch.device("cpu"), discrete_action=True
    )
    obs = torch.randn(8, 6)
    action = torch.randint(0, 8, (8, 1))
    reward = torch.randn(8)
    next_obs = torch.randn(8, 6)
    done = torch.zeros(8)
    buf.add(obs, action, reward, next_obs, done, behaviour_logprob=None, latent=None)
    batch = buf.sample(4)
    assert batch.obs.shape == (4, 6)


def test_csv_logger_contract(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    logger = CSVLogger(path, ["a", "b"])
    logger.write({"a": 1})
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["a", "b"]
    assert rows[1] == ["1", ""]


def test_offline_collector_runs() -> None:
    env = make("tabular-sepsis-v0", cell=3, n_envs=8, device="cpu")
    from causal_rl.behaviour.uniform import UniformExplorer

    pol = UniformExplorer(n_actions=8)
    buffer = collect_offline_dataset(
        env=env,
        behaviour_policy=pol,
        n_transitions=64,
        expose_pi_b=True,
        expose_latent=False,
        seed=0,
    )
    assert buffer.size == 64


def test_online_runner_path(tmp_path: Path) -> None:
    cfg = RunnerConfig(
        cell=1,
        env_name="tabular-sepsis-v0",
        algorithm="ppo",
        behaviour="uniform",
        seed=0,
        total_frames=20,
        n_checkpoints_train=2,
        n_checkpoints_eval=2,
        output_dir=tmp_path / "online",
        device="cpu",
        n_envs=8,
        batch_size=8,
        offline_transitions=64,
        offline_updates=10,
    )
    BenchmarkRunner(cfg).run()
    assert (cfg.output_dir / "train.csv").exists()


def test_utils_basic() -> None:
    rs = set_seed(0)
    assert rs.torch_seed >= 0
    assert str(get_device("cpu")) == "cpu"
    assert isinstance(get_git_commit(), str)
