from __future__ import annotations

import torch

from causal_rl.behaviour.reward_aligned import RewardAlignedExplorer
from causal_rl.behaviour.reward_misaligned import RewardMisalignedExplorer


def test_reward_misaligned_requires_latent() -> None:
    pol = RewardMisalignedExplorer(n_actions=4, beta=1.0, requires_latent=True)
    obs = torch.randn(8, 6)
    try:
        pol.select_action(obs, latent=None)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_reward_misaligned_has_explicit_family_name() -> None:
    assert RewardMisalignedExplorer.family == "reward_misaligned"


def test_reward_misaligned_discrete_is_anti_aligned() -> None:
    obs = torch.full((4096, 6), 0.8)
    aligned = RewardAlignedExplorer(n_actions=8, beta=3.0)
    misaligned = RewardMisalignedExplorer(n_actions=8, beta=3.0, bias_strength=0.0)
    a_good, _ = aligned.select_action(obs)
    a_bad, _ = misaligned.select_action(obs)
    assert a_bad.float().mean().item() < a_good.float().mean().item()


def test_reward_misaligned_continuous_is_anti_aligned() -> None:
    aligned = RewardAlignedExplorer(act_dim=3, beta=2.0)
    misaligned = RewardMisalignedExplorer(act_dim=3, beta=2.0, bias_strength=1.0)
    high_obs = torch.full((2048, 6), 0.9)
    low_obs = torch.full((2048, 6), 0.1)
    torch.manual_seed(11)
    aligned_high, _ = aligned.select_action(high_obs)
    torch.manual_seed(11)
    aligned_low, _ = aligned.select_action(low_obs)
    torch.manual_seed(11)
    misaligned_high, _ = misaligned.select_action(high_obs)
    torch.manual_seed(11)
    misaligned_low, _ = misaligned.select_action(low_obs)
    assert aligned_high.mean().item() > aligned_low.mean().item()
    assert misaligned_high.mean().item() < misaligned_low.mean().item()


def test_reward_misaligned_continuous_action_support() -> None:
    pol = RewardMisalignedExplorer(act_dim=3, beta=1.0)
    obs = torch.randn(16, 6)
    action, log_prob = pol.select_action(obs)
    assert action.shape == (16, 3)
    assert log_prob.shape == (16,)
