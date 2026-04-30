from __future__ import annotations

import torch

from causal_rl.behaviour.curiosity import CuriosityExplorer
from causal_rl.behaviour.novelty_trap import NoveltyTrapExplorer


def test_novelty_trap_requires_latent() -> None:
    pol = NoveltyTrapExplorer(n_actions=4, delta=1.0, requires_latent=True)
    obs = torch.randn(8, 6)
    try:
        pol.select_action(obs, latent=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_novelty_trap_discrete_biases_risky_actions() -> None:
    obs = torch.randn(4096, 6)
    trap = NoveltyTrapExplorer(n_actions=8, delta=0.0, risk_strength=4.0, epsilon_mix=0.0)
    base = CuriosityExplorer(n_actions=8, delta=0.0)
    a_trap, _ = trap.select_action(obs)
    a_base, _ = base.select_action(obs)
    assert a_trap.float().mean().item() > a_base.float().mean().item()


def test_novelty_trap_continuous_action_support() -> None:
    pol = NoveltyTrapExplorer(act_dim=3, delta=1.0)
    obs = torch.randn(16, 6)
    action, log_prob = pol.select_action(obs)
    assert action.shape == (16, 3)
    assert log_prob.shape == (16,)
