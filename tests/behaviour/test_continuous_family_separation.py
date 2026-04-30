from __future__ import annotations

import torch

from causal_rl.behaviour.curiosity import CuriosityExplorer
from causal_rl.behaviour.information import InformationExplorer
from causal_rl.behaviour.reward_aligned import RewardAlignedExplorer
from causal_rl.behaviour.reward_misaligned import RewardMisalignedExplorer


def test_continuous_behaviour_families_are_explicitly_separated() -> None:
    obs = torch.linspace(0.0, 1.0, steps=4096 * 6, dtype=torch.float32).reshape(4096, 6)
    policies = {
        "aligned": RewardAlignedExplorer(act_dim=3, beta=2.0),
        "misaligned": RewardMisalignedExplorer(act_dim=3, beta=2.0, bias_strength=1.0),
        "information": InformationExplorer(act_dim=3, gamma_info=2.0),
        "curiosity": CuriosityExplorer(act_dim=3, delta=2.0),
    }
    actions: dict[str, torch.Tensor] = {}
    for name, policy in policies.items():
        # Reset seed so pairwise differences are not explained by different noise draws.
        torch.manual_seed(123)
        action, _ = policy.select_action(obs)
        actions[name] = action

    names = sorted(actions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = (actions[names[i]] - actions[names[j]]).abs().mean().item()
            assert d > 0.08, f"continuous policies too similar: {names[i]} vs {names[j]} (MAD={d:.4f})"
