from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from causal_rl.behaviour.base import BiasedExplorer
from causal_rl.behaviour.certainty_seeking import CertaintySeekingExplorer
from causal_rl.behaviour.curiosity import CuriosityExplorer
from causal_rl.behaviour.information import InformationExplorer
from causal_rl.behaviour.novelty_trap import NoveltyTrapExplorer
from causal_rl.behaviour.reward_aligned import RewardAlignedExplorer
from causal_rl.behaviour.reward_misaligned import RewardMisalignedExplorer
from causal_rl.behaviour.uniform import UniformExplorer


@dataclass(frozen=True)
class BehaviourSpec:
    name: str
    builder: Callable[..., BiasedExplorer]


BEHAVIOURS: dict[str, BehaviourSpec] = {}


def register(spec: BehaviourSpec) -> None:
    BEHAVIOURS[spec.name] = spec


def make(name: str, **kwargs: object) -> BiasedExplorer:
    if name not in BEHAVIOURS:
        msg = f"Unknown behaviour policy: {name}"
        raise KeyError(msg)
    return BEHAVIOURS[name].builder(**kwargs)


register(BehaviourSpec("uniform", UniformExplorer))
register(BehaviourSpec("reward_aligned", RewardAlignedExplorer))
register(BehaviourSpec("reward_misaligned", RewardMisalignedExplorer))
register(BehaviourSpec("information", InformationExplorer))
register(BehaviourSpec("certainty_seeking", CertaintySeekingExplorer))
register(BehaviourSpec("curiosity", CuriosityExplorer))
register(BehaviourSpec("novelty_trap", NoveltyTrapExplorer))
