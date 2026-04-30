from __future__ import annotations

import torch

from causal_rl.behaviour.uniform import UniformExplorer


def test_uniform_explorer_discrete() -> None:
    pol = UniformExplorer(n_actions=8)
    obs = torch.randn(256, 6)
    action, log_prob = pol.select_action(obs)
    assert action.shape == (256, 1)
    assert log_prob.shape == (256,)
