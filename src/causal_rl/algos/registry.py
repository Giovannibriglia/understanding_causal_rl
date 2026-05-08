from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from causal_rl.algos.base import BaseAlgorithm
from causal_rl.algos.off_policy.bound_cql import BoundCQL
from causal_rl.algos.off_policy.confounded_dqn import ConfoundedDQN
from causal_rl.algos.off_policy.cql import CQL
from causal_rl.algos.off_policy.ddpg import DDPG
from causal_rl.algos.off_policy.dqn import DQN
from causal_rl.algos.off_policy.offline_bandit_naive import OfflineBanditNaive
from causal_rl.algos.on_policy.a2c import A2C
from causal_rl.algos.on_policy.ppo import PPO
from causal_rl.algos.on_policy.rct import RCT
from causal_rl.algos.on_policy.ucb import UCB
from causal_rl.algos.on_policy.ucb_minus import UCBMinus
from causal_rl.algos.on_policy.ucb_plus import UCBPlus


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    builder: Callable[..., BaseAlgorithm]
    action_type: Literal["discrete", "continuous", "both"]
    supported_cells: set[int]
    kind: Literal["on_policy", "off_policy"]
    # Preferred parallelism when n_envs is not set in the YAML config.
    default_n_envs: int = 64


ALGOS: dict[str, AlgoSpec] = {}


def register(spec: AlgoSpec) -> None:
    ALGOS[spec.name] = spec


def make(name: str, **kwargs: object) -> BaseAlgorithm:
    if name not in ALGOS:
        msg = f"Unknown algorithm: {name}"
        raise KeyError(msg)
    return ALGOS[name].builder(**kwargs)


_ALL_CELLS: set[int] = {1, 2, 3, 4}

# PPO / A2C are rollout-bound — large batches converge faster.
register(AlgoSpec("ppo", PPO, "discrete", _ALL_CELLS, "on_policy", default_n_envs=256))
register(AlgoSpec("a2c", A2C, "discrete", _ALL_CELLS, "on_policy", default_n_envs=256))
# DQN / CQL / DDPG / BoundCQL are memory-light; 64 is ample.
register(AlgoSpec("dqn", DQN, "discrete", _ALL_CELLS, "off_policy", default_n_envs=64))
register(AlgoSpec("ddpg", DDPG, "continuous", _ALL_CELLS, "off_policy", default_n_envs=64))
register(AlgoSpec("cql", CQL, "discrete", _ALL_CELLS, "off_policy", default_n_envs=64))
register(
    AlgoSpec(
        "confounded_dqn", ConfoundedDQN, "discrete", _ALL_CELLS, "off_policy", default_n_envs=64
    )
)
register(AlgoSpec("bound_cql", BoundCQL, "discrete", _ALL_CELLS, "off_policy", default_n_envs=64))
# UCB-family maintains per-arm counts — tiny batches are sufficient.
register(AlgoSpec("ucb", UCB, "discrete", _ALL_CELLS, "on_policy", default_n_envs=8))
register(AlgoSpec("ucb_minus", UCBMinus, "discrete", _ALL_CELLS, "on_policy", default_n_envs=8))
register(AlgoSpec("ucb_plus", UCBPlus, "discrete", _ALL_CELLS, "on_policy", default_n_envs=8))
register(AlgoSpec("rct", RCT, "discrete", _ALL_CELLS, "on_policy", default_n_envs=8))
# v21: offline-bandit estimators (per-arm running mean ± IPW) for the
# off-policy bandit family.  Bandit-default n_envs (8) — these are
# tiny estimators, no benefit from large batches.
register(
    AlgoSpec(
        "offline_bandit_naive", OfflineBanditNaive, "discrete", _ALL_CELLS,
        "off_policy", default_n_envs=8,
    )
)
