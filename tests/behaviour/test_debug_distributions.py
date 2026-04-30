from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from causal_rl.behaviour.reward_aligned import RewardAlignedExplorer


def test_debug_action_distribution_plot(tmp_path: Path) -> None:
    obs = torch.randn(1024, 6)
    betas = [0.0, 0.5, 1.0, 2.0]
    fig, ax = plt.subplots()
    for beta in betas:
        pol = RewardAlignedExplorer(n_actions=4, beta=beta)
        act, _ = pol.select_action(obs)
        probs = torch.bincount(act.squeeze(-1), minlength=4).float()
        probs = probs / probs.sum()
        ax.plot([0, 1, 2, 3], probs.numpy(), label=f"beta={beta}")
    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.legend()
    out = tmp_path / "appendix_c_reward_aligned.pdf"
    fig.savefig(out)
    assert out.exists()
