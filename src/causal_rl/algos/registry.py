from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from causal_rl.algos.base import BaseAlgorithm
from causal_rl.algos.off_policy.confounded_dqn import ConfoundedDQN
from causal_rl.algos.off_policy.cql import CQL
from causal_rl.algos.off_policy.ddpg import DDPG
from causal_rl.algos.off_policy.dqn import DQN
from causal_rl.algos.on_policy.a2c import A2C
from causal_rl.algos.on_policy.ppo import PPO


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    builder: Callable[..., BaseAlgorithm]
    action_type: Literal["discrete", "continuous", "both"]
    supported_cells: set[int]
    kind: Literal["on_policy", "off_policy"]


ALGOS: dict[str, AlgoSpec] = {}


def register(spec: AlgoSpec) -> None:
    ALGOS[spec.name] = spec


def make(name: str, **kwargs: object) -> BaseAlgorithm:
    if name not in ALGOS:
        msg = f"Unknown algorithm: {name}"
        raise KeyError(msg)
    return ALGOS[name].builder(**kwargs)


_ALL_CELLS: set[int] = {1, 2, 3, 4}

register(AlgoSpec("ppo", PPO, "discrete", _ALL_CELLS, "on_policy"))
register(AlgoSpec("a2c", A2C, "discrete", _ALL_CELLS, "on_policy"))
register(AlgoSpec("dqn", DQN, "discrete", _ALL_CELLS, "off_policy"))
register(AlgoSpec("ddpg", DDPG, "continuous", _ALL_CELLS, "off_policy"))
register(AlgoSpec("cql", CQL, "discrete", _ALL_CELLS, "off_policy"))
register(AlgoSpec("confounded_dqn", ConfoundedDQN, "discrete", _ALL_CELLS, "off_policy"))
