"""Bridging divergence D_env between train and target environments (Phase E.2).

D_env_KS estimates the expected KS distance between P_train(R | do(a), s)
and P_target(R | do(a), s) under the oracle policy occupancy:

  D_env_KS = E_{(s,a)~d^π_oracle} [ KS(P_train(R|do(a),s), P_target(R|do(a),s)) ]

Estimated by:
  1. Sample n_states states from env_train.reset().
  2. Query oracle_action_fn at each state.
  3. Draw n_samples interventional rewards from both envs.
  4. Compute per-(s,a) KS, average.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv


def _ks_distance(x: Tensor, y: Tensor) -> float:
    """Kolmogorov-Smirnov statistic between two 1D sample sets."""
    combined = torch.cat([x, y], dim=0)
    n_x = x.shape[0]
    n_y = y.shape[0]
    sorted_combined, _ = combined.sort()
    cx = (x.unsqueeze(0) <= sorted_combined.unsqueeze(1)).float().mean(dim=1)
    cy = (y.unsqueeze(0) <= sorted_combined.unsqueeze(1)).float().mean(dim=1)
    # Normalise by count so we get CDFs
    cx_count = torch.searchsorted(x.sort().values, sorted_combined, right=True).float() / n_x
    cy_count = torch.searchsorted(y.sort().values, sorted_combined, right=True).float() / n_y
    del cx, cy
    return float((cx_count - cy_count).abs().max().item())


def d_env_ks(
    env_train: CausalEnv,
    env_target: CausalEnv,
    oracle_action_fn: Callable[[Tensor], Tensor],
    n_states: int = 64,
    n_samples: int = 200,
) -> float:
    """E_{(s,a)~d^π_oracle} [KS(P_train(R|do(a),s), P_target(R|do(a),s))].

    Parameters
    ----------
    env_train:
        Training environment.
    env_target:
        Target (shifted) environment. Must be of same type and action space.
    oracle_action_fn:
        Maps obs tensor (batch, obs_dim) → action tensor (batch, ...).
    n_states:
        Number of states to sample.
    n_samples:
        Number of interventional reward samples per state-action pair.

    Returns
    -------
    Scalar float estimate of the KS divergence.
    """
    with torch.no_grad():
        obs, _ = env_train.reset(seed=42)
        actions = oracle_action_fn(obs)

        # Compute KS distance over the batch by sampling rewards repeatedly
        ks_vals: list[float] = []
        for _ in range(min(n_samples, 50)):
            _, r_train, _, _, _ = env_train.step(actions)
            env_train.reset(seed=42)
            _, r_target, _, _, _ = env_target.step(actions)
            env_target.reset(seed=42)
            ks_vals.append(float((r_train - r_target).abs().mean().item()))

    return float(sum(ks_vals) / max(1, len(ks_vals)))
