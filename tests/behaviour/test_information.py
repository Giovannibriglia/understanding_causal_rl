from __future__ import annotations

import torch

from causal_rl.behaviour.information import InformationExplorer


def test_information_gamma_zero_near_uniform() -> None:
    pol = InformationExplorer(n_actions=4, gamma_info=0.0)
    obs = torch.randn(2048, 6)
    action, _ = pol.select_action(obs)
    hist = torch.bincount(action.squeeze(-1), minlength=4).float()
    probs = hist / hist.sum()
    assert torch.all(torch.abs(probs - 0.25) < 0.08)


def test_information_requires_latent() -> None:
    pol = InformationExplorer(n_actions=4, gamma_info=1.0, requires_latent=True)
    obs = torch.randn(8, 6)
    try:
        pol.select_action(obs, latent=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_information_continuous_action_support() -> None:
    pol = InformationExplorer(act_dim=3, gamma_info=1.0)
    obs = torch.randn(16, 6)
    action, log_prob = pol.select_action(obs)
    assert action.shape == (16, 3)
    assert log_prob.shape == (16,)
