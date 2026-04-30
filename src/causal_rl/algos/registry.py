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


register(AlgoSpec("ppo", PPO, "discrete", {1, 2}, "on_policy"))
register(AlgoSpec("a2c", A2C, "discrete", {1, 2}, "on_policy"))
register(AlgoSpec("dqn", DQN, "discrete", {1, 2, 3, 4, 5, 6, 7, 8}, "off_policy"))
register(AlgoSpec("ddpg", DDPG, "continuous", {1, 2, 3, 4, 5, 6, 7, 8}, "off_policy"))
register(AlgoSpec("cql", CQL, "discrete", {3, 4, 5, 6, 7, 8}, "off_policy"))
register(
    AlgoSpec(
        "confounded_dqn",
        ConfoundedDQN,
        "discrete",
        {3, 4, 5, 6, 7, 8},
        "off_policy",
    )
)
