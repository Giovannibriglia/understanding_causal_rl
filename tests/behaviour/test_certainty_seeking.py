from __future__ import annotations

import torch

from causal_rl.behaviour.certainty_seeking import CertaintySeekingExplorer
from causal_rl.behaviour.information import InformationExplorer


def test_certainty_seeking_requires_latent() -> None:
    pol = CertaintySeekingExplorer(n_actions=4, gamma_info=1.0, requires_latent=True)
    obs = torch.randn(8, 6)
    try:
        pol.select_action(obs, latent=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_certainty_seeking_discrete_prefers_lower_index_than_information() -> None:
    obs = torch.randn(4096, 6) * 2.0
    info_pol = InformationExplorer(n_actions=8, gamma_info=2.0)
    cert_pol = CertaintySeekingExplorer(n_actions=8, gamma_info=2.0)
    a_info, _ = info_pol.select_action(obs)
    a_cert, _ = cert_pol.select_action(obs)
    assert a_cert.float().mean().item() < a_info.float().mean().item()


def test_certainty_seeking_continuous_action_support() -> None:
    pol = CertaintySeekingExplorer(act_dim=3, gamma_info=1.0)
    obs = torch.randn(16, 6)
    action, log_prob = pol.select_action(obs)
    assert action.shape == (16, 3)
    assert log_prob.shape == (16,)
