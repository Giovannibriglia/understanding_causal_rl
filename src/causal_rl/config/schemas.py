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


class CoverageRegime(BaseModel):
    """One entry in ``MatrixConfig.coverage_regimes``.

    Three knobs deliberately combined to produce a *coverage regime*:
    full data, small data, biased behaviour, or action-masked.  Each
    regime produces its own batch of runs in the matrix expansion.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    offline_transitions: int = 8000
    forbidden_actions: list[int] = Field(default_factory=list)
    bias_strength: float = 1.0
    n_action_restriction_steps: int | None = None


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
    n_checkpoints_train: int = Field(
        50, ge=2, description="Min 2; always include first and last."
    )
    n_checkpoints_eval: int = Field(
        10, ge=2, description="Min 2; always include first and last."
    )
    horizon: int | None = None
    rollout_horizon: int | None = None
    output: str
    n_envs: int = 64
    batch_size: int = 64
    offline_transitions: int = 50_000
    offline_updates: int = 2_000
    # n_eval_episodes × n_envs is the effective sample size for the
    # perturbed-return mean.  3 × 16 = 48 trajectories is sufficient at
    # smoke budgets; bump to 5+ for paper runs by overriding in the YAML.
    n_eval_episodes: int = Field(3, ge=1)
    # Optional: trim the perturbation grid for smoke configs.  None ⇒ full
    # grid; <= 0 also means full.  See ``runner._evaluate_perturbations``.
    perturbation_grid_size: int | None = None
    eval_perturbations: bool = True
    # v11 coverage-stress knobs.  ``forbidden_actions`` blocks the listed
    # action indices from the offline buffer; ``n_action_restriction_steps``
    # gates the restriction to the first k collected transitions.  Both
    # default to no restriction.
    forbidden_actions: list[int] | None = None
    n_action_restriction_steps: int | None = None
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
    total_frames_per_episode: int | None = None
    n_checkpoints_train: int = Field(
        50, ge=2, description="Min 2; always include first and last."
    )
    n_checkpoints_eval: int = Field(
        10, ge=2, description="Min 2; always include first and last."
    )
    env_horizons: dict[str, int] = Field(default_factory=dict)
    horizon_sweep: list[int] | None = None
    offline_transitions: int = 50_000
    offline_updates: int = 2_000
    alpha_conf: float = 0.0
    alpha_conf_sweep: list[float] | None = None
    bias_strengths: list[float] | None = None
    offline_transitions_sweep: list[int] | None = None
    n_envs: int = 64
    eval_n_envs: int | None = None
    n_eval_episodes: int = Field(3, ge=1)
    # Optional: trim the perturbation grid for smoke configs.
    perturbation_grid_size: int | None = None
    eval_perturbations: bool = True
    # v11 coverage-stress sweep axis.  Each entry is one regime; the
    # matrix runner emits one task per (cell × algo × … × regime).  Each
    # regime overrides ``offline_transitions``, ``forbidden_actions``, and
    # ``bias_strength`` for the runs it produces.
    coverage_regimes: list[CoverageRegime] | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"YAML root must be mapping: {path}"
        raise ValueError(msg)
    return data
