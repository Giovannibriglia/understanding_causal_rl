from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class EnvOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_envs: int | None = None
    horizon: int | None = None
    alpha_conf: float | None = None


class AlgoOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: float | None = None
    gamma: float | None = None
    batch_size: int | None = None


class RunConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cell: int
    env: str
    algorithm: str
    behaviour: str
    seed: int
    total_frames: int = 50_000
    n_checkpoints_train: int = 50
    n_checkpoints_eval: int = 10
    horizon: int | None = None
    rollout_horizon: int | None = None
    output: str
    n_envs: int = 64
    batch_size: int = 64
    offline_transitions: int = 50_000
    offline_updates: int = 2_000
    env_overrides: EnvOverrides = Field(default_factory=EnvOverrides)
    algo_overrides: AlgoOverrides = Field(default_factory=AlgoOverrides)


class MatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cells: list[int]
    envs: list[str]
    algorithms: list[str]
    behaviours: list[str]
    seeds: list[int] | None = None
    total_frames: int = 20_000
    n_checkpoints_train: int = 50
    n_checkpoints_eval: int = 10
    env_horizons: dict[str, int] = Field(default_factory=dict)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"YAML root must be mapping: {path}"
        raise ValueError(msg)
    return data
