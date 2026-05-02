"""Bridging divergence D_env between train and target environments (Phase E.2).

D_env_KS estimates the expected KS distance between P_train(R | do(a), s)
and P_target(R | do(a), s) under the oracle policy occupancy:

  D_env_KS = E_{(s,a)~d^π_oracle} [ KS(P_train(R|do(a),s), P_target(R|do(a),s)) ]

Estimated by:
  1. Reset both envs to the same seed to sync states.
  2. Query oracle_action_fn at each state.
  3. Draw n_samples interventional rewards from both envs via sample_interventional.
  4. Compute per-(s,a) KS statistic, average.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from causal_rl.envs.base import CausalEnv


def _ks_distance(x: Tensor, y: Tensor) -> float:
    """Two-sample KS statistic between empirical CDFs."""
    grid = torch.cat([x, y]).sort().values
    cdf_x = torch.searchsorted(x.sort().values, grid, right=True).float() / x.numel()
    cdf_y = torch.searchsorted(y.sort().values, grid, right=True).float() / y.numel()
    return float((cdf_x - cdf_y).abs().max().item())


def d_env_ks(
    env_train: CausalEnv,
    env_target: CausalEnv,
    oracle_action_fn: Callable[[Tensor], Tensor],
    n_states: int = 64,
    n_samples: int = 200,
) -> float:
    """E_{(s,a) ~ d^π_oracle} [ KS( P_train(R | do(a), s),  P_target(R | do(a), s) ) ].

    Parameters
    ----------
    env_train:
        Training environment.
    env_target:
        Target (shifted) environment. Must be of same type and action space.
    oracle_action_fn:
        Maps obs tensor (batch, obs_dim) → action tensor (batch, ...).
    n_states:
        Number of parallel state-action pairs to evaluate.
    n_samples:
        Number of interventional reward samples per state-action pair.

    Returns
    -------
    Scalar float estimate of the expected KS divergence.
    """
    with torch.no_grad():
        # Sync states: reset both envs to the same seed.
        obs, _ = env_train.reset(seed=42)
        env_target.reset(seed=42)
        actions = oracle_action_fn(obs)

        n = min(obs.shape[0], n_states)
        obs_n = obs[:n]
        acts_n = actions[:n]

        # Draw n_samples interventional rewards from each env at the synced states.
        r_train = env_train.sample_interventional(obs_n, acts_n, n=n_samples)  # (n, n_samples)
        r_target = env_target.sample_interventional(obs_n, acts_n, n=n_samples)  # (n, n_samples)

        ks_vals = [_ks_distance(r_train[i], r_target[i]) for i in range(n)]

    return float(sum(ks_vals) / max(1, len(ks_vals)))
