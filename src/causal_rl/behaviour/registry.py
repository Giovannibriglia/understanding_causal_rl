from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from causal_rl.behaviour.base import BiasedExplorer
from causal_rl.behaviour.certainty_seeking import CertaintySeekingExplorer
from causal_rl.behaviour.curiosity import CuriosityExplorer
from causal_rl.behaviour.information import InformationExplorer
from causal_rl.behaviour.novelty_trap import NoveltyTrapExplorer
from causal_rl.behaviour.reward_aligned import RewardAlignedExplorer
from causal_rl.behaviour.reward_aligned_z import RewardAlignedZExplorer
from causal_rl.behaviour.reward_misaligned import RewardMisalignedExplorer
from causal_rl.behaviour.uniform import UniformExplorer


@dataclass(frozen=True)
class BehaviourSpec:
    name: str
    builder: Callable[..., BiasedExplorer]
    depends_on_u: bool


BEHAVIOURS: dict[str, BehaviourSpec] = {}


def register(spec: BehaviourSpec) -> None:
    BEHAVIOURS[spec.name] = spec


def make(name: str, **kwargs: object) -> BiasedExplorer:
    if name not in BEHAVIOURS:
        msg = f"Unknown behaviour policy: {name}"
        raise KeyError(msg)
    return BEHAVIOURS[name].builder(**kwargs)


register(BehaviourSpec("uniform", UniformExplorer, depends_on_u=False))
register(BehaviourSpec("reward_aligned", RewardAlignedExplorer, depends_on_u=True))
# v26: Z-peeking variant of reward_aligned.  Distinct registry entry so
# YAML configs are explicit about which behaviour they exercise.  Old
# ``reward_aligned`` is preserved as the U-peeking baseline (used by
# ``coverage_stress.yaml``); new ``reward_aligned_z`` is used by
# paper.yaml / paper_bandit_offpolicy.yaml / smoke_v8.yaml.
register(BehaviourSpec("reward_aligned_z", RewardAlignedZExplorer, depends_on_u=True))
register(BehaviourSpec("reward_misaligned", RewardMisalignedExplorer, depends_on_u=True))
register(BehaviourSpec("information", InformationExplorer, depends_on_u=False))
register(BehaviourSpec("certainty_seeking", CertaintySeekingExplorer, depends_on_u=False))
register(BehaviourSpec("curiosity", CuriosityExplorer, depends_on_u=False))
register(BehaviourSpec("novelty_trap", NoveltyTrapExplorer, depends_on_u=False))
