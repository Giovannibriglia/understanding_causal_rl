from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from causal_rl.envs.bandit_view import BanditView
from causal_rl.envs.base import CausalEnv
from causal_rl.envs.continuous_ward import ContinuousWardEnv
from causal_rl.envs.tabular_sepsis import TabularSepsisEnv


@dataclass(frozen=True)
class EnvSpec:
    name: str
    builder: Callable[..., CausalEnv]
    fidelity: Literal["tabular", "continuous"]


ENVS: dict[str, EnvSpec] = {}


def register(spec: EnvSpec) -> None:
    ENVS[spec.name] = spec


def make(name: str, cell: int, **kwargs: object) -> CausalEnv:
    if name not in ENVS:
        msg = f"Unknown env: {name}"
        raise KeyError(msg)
    return ENVS[name].builder(cell=cell, **kwargs)


register(EnvSpec(name="tabular-sepsis-v0", builder=TabularSepsisEnv, fidelity="tabular"))
register(EnvSpec(name="continuous-ward-v0", builder=ContinuousWardEnv, fidelity="continuous"))


def _make_bandit_tabular_sepsis(
    cell: int, n_envs: int = 64, device: str | None = None, **kwargs: object
) -> CausalEnv:
    env = TabularSepsisEnv(cell, n_envs, device=device, **kwargs)  # type: ignore[arg-type]
    return BanditView(env)


register(
    EnvSpec(
        name="bandit-tabular-sepsis-v0",
        builder=_make_bandit_tabular_sepsis,
        fidelity="tabular",
    )
)
